# train_bev.py
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import (DATA_ROOT, CKPT_DIR, OUTPUT_DIR,
                    NUM_CLASSES, BEV_SIZE, BEV_RANGE,
                    CAT_MAP, CAM_CHANNEL)
from bev.lss import LSSSegmentor
from utils.visualize import plot_training_curves


class NuScenesBEVDataset(Dataset):
    """
    返回：
        image:     (3, H, W) 归一化图像
        intrinsic: (3, 3)    相机内参
        bev_mask:  (bev_size, bev_size) IPM 投影的 BEV 语义标注
    """
    def __init__(self, nusc, split='train', image_size=(224, 400)):
        self.nusc       = nusc
        self.image_size = image_size
        # Scene-level split，和 nuscenes_seg.py 保持一致
        n_scenes  = len(nusc.scene)
        split_idx = int(n_scenes * 0.8)
        train_scenes = set(s['token'] for s in nusc.scene[:split_idx])
        val_scenes   = set(s['token'] for s in nusc.scene[split_idx:])

        def get_scene_token(sample):
            return nusc.get('scene', sample['scene_token'])['token']

        all_samples = nusc.sample
        if split == 'train':
            self.samples = [s for s in all_samples
                            if get_scene_token(s) in train_scenes]
        else:
            self.samples = [s for s in all_samples
                            if get_scene_token(s) in val_scenes]
        print(f"[BEVDataset] {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from pyquaternion import Quaternion
        from nuscenes.utils.geometry_utils import view_points
        import numpy as np

        sample   = self.samples[idx]
        cam_token = sample['data'][CAM_CHANNEL]
        cam_data = self.nusc.get('sample_data', cam_token)
        cs = self.nusc.get('calibrated_sensor',
                           cam_data['calibrated_sensor_token'])
        ego = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        # 图像
        img = Image.open(DATA_ROOT + '/' + cam_data['filename']).convert('RGB')
        W0, H0 = img.size
        img = img.resize((self.image_size[1], self.image_size[0]))
        img_tensor = torch.from_numpy(
            np.array(img).transpose(2, 0, 1)).float() / 255.0

        # 内参（随图像缩放调整）
        intrinsic = np.array(cs['camera_intrinsic'], dtype=np.float32)
        intrinsic[0] *= self.image_size[1] / W0
        intrinsic[1] *= self.image_size[0] / H0
        
        # 外参：camera-to-ego 变换矩阵 (4x4)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
        extrinsic[:3,  3] = np.array(cs['translation'])

        # 生成 BEV mask（用 IPM 投影）
        from bev.ipm import ipm_semantic
        bev_mask = self._make_bev_mask(
            self.nusc, sample, cs, ego, H0, W0, intrinsic)

        # 生成 sparse depth label
        from utils.depth_utils import lidar_to_sparse_depth, depth_to_bin_label
        depth_map = lidar_to_sparse_depth(
            self.nusc, sample['token'], CAM_CHANNEL,
            image_size=(H0, W0))
        depth_label = depth_to_bin_label(depth_map)
        # resize 到模型特征图尺寸 (H/16, W/16)
        feat_h = self.image_size[0] // 16
        feat_w = self.image_size[1] // 16
        from utils.depth_utils import resize_depth
        depth_label = resize_depth(depth_label, (feat_h, feat_w))

        return {
            'image':       img_tensor,
            'intrinsic':   torch.from_numpy(intrinsic),
            'extrinsic':   torch.from_numpy(extrinsic),
            'bev_mask':    torch.from_numpy(bev_mask).long(),
            'depth_label': torch.from_numpy(depth_label).long(),
        }

    def _make_bev_mask(self, nusc, sample, cs, ego, H, W, intrinsic):
        from pyquaternion import Quaternion
        from nuscenes.utils.geometry_utils import view_points

        # 先生成 2D mask
        mask_2d = np.zeros((H, W), dtype=np.uint8)
        for ann_token in sample['anns']:
            ann   = nusc.get('sample_annotation', ann_token)
            label = 0
            for key, val in CAT_MAP.items():
                if ann['category_name'].startswith(key):
                    label = val
                    break
            if label == 0:
                continue
            box = nusc.get_box(ann_token)
            box.translate(-np.array(ego['translation']))
            box.rotate(Quaternion(ego['rotation']).inverse)
            box.translate(-np.array(cs['translation']))
            box.rotate(Quaternion(cs['rotation']).inverse)
            if box.center[2] < 0.1:
                continue
            corners = view_points(
                box.corners(),
                np.array(cs['camera_intrinsic']),
                normalize=True)
            x1 = int(np.clip(corners[0].min(), 0, W))
            x2 = int(np.clip(corners[0].max(), 0, W))
            y1 = int(np.clip(corners[1].min(), 0, H))
            y2 = int(np.clip(corners[1].max(), 0, H))
            if x2 > x1 and y2 > y1:
                mask_2d[y1:y2, x1:x2] = label

        # 用 IPM 投影到 BEV
        from bev.ipm import ipm_semantic
        bev_mask = ipm_semantic(mask_2d, intrinsic)
        return bev_mask


def compute_miou(preds, labels, num_classes=NUM_CLASSES):
    ious = []
    preds  = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    for cls in range(num_classes):
        pred_cls  = preds  == cls
        label_cls = labels == cls
        inter = (pred_cls & label_cls).sum()
        union = (pred_cls | label_cls).sum()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def train():
    from nuscenes.nuscenes import NuScenes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    nusc = NuScenes(version='v1.0-mini',
                    dataroot=DATA_ROOT, verbose=False)

    train_dataset = NuScenesBEVDataset(nusc, split='train')
    val_dataset   = NuScenesBEVDataset(nusc, split='val')
    train_loader  = DataLoader(train_dataset, batch_size=4,
                               shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=4,
                               shuffle=False, num_workers=2)

    model     = LSSSegmentor(num_classes=NUM_CLASSES,
                             bev_size=BEV_SIZE,
                             bev_range=BEV_RANGE).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)

    ckpt_path = f'{CKPT_DIR}/lss_best.pth'
    best_miou = 0.0
    # if os.path.exists(ckpt_path):
    #     model.load_state_dict(torch.load(ckpt_path, map_location=device))
    #     print(f"加载已有权重：{ckpt_path}")

    train_losses, val_mious, val_epochs = [], [], []
    NUM_EPOCHS = 30

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            images      = batch['image'].to(device)
            intrinsics  = batch['intrinsic'].to(device)
            extrinsics  = batch['extrinsic'].to(device)
            bev_masks   = batch['bev_mask'].to(device)

            logits, depth_probs = model(images, intrinsics,
                                        extrinsics, return_depth=True)
            # BEV 分割 loss
            weights = torch.tensor([0.1, 2.0, 3.0, 3.0, 2.0],
                                    device=device)
            bev_loss = F.cross_entropy(logits, bev_masks, weight=weights)

            # Sparse depth supervision loss
            depth_labels = batch['depth_label'].to(device)
            valid = depth_labels > 0
            if valid.sum() > 0:
                depth_log = depth_probs.log().permute(0, 2, 3, 1)
                depth_tgt = (depth_labels - 1).clamp(0)
                depth_loss = F.nll_loss(
                    depth_log[valid].unsqueeze(0).permute(0, 2, 1),
                    depth_tgt[valid].unsqueeze(0),
                    reduction='mean'
                )
                loss = bev_loss + 0.1 * depth_loss
            else:
                loss = bev_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            model.eval()
            all_mious = []
            with torch.no_grad():
                for batch in val_loader:
                    images      = batch['image'].to(device)
                    intrinsics  = batch['intrinsic'].to(device)
                    extrinsics  = batch['extrinsic'].to(device)
                    bev_masks   = batch['bev_mask'].to(device)
                    logits      = model(images, intrinsics, extrinsics)
                    preds      = logits.argmax(dim=1)
                    for p, l in zip(preds, bev_masks):
                        all_mious.append(compute_miou(p, l))

            val_miou = np.mean(all_mious)
            val_mious.append(val_miou)
            val_epochs.append(epoch + 1)
            train_losses.append(avg_loss)

            if val_miou > best_miou:
                best_miou = val_miou
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

            print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val mIoU: {val_miou:.4f} | "
                  f"Best: {best_miou:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training_curves(
        train_losses, val_mious, val_epochs,
        save_path=f'{OUTPUT_DIR}/bev_training_curves.png'
    )
    print(f"\nBEV 训练完成！最佳 mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    train()