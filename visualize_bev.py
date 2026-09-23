# visualize_bev.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import (DATA_ROOT, CKPT_DIR, OUTPUT_DIR,
                    NUM_CLASSES, BEV_SIZE, BEV_RANGE,
                    SEG_COLORS, ID2LABEL, CAM_CHANNEL)
from bev.lss import LSSSegmentor
from bev.ipm import ipm_transform, ipm_semantic
from models.segformer import get_processor, load_trained_model


COLORS = np.array(
    [SEG_COLORS[i] for i in range(len(SEG_COLORS))],
    dtype=np.uint8
)


def visualize_ipm_vs_lss(nusc, num_samples=4):
    """
    对比可视化：IPM baseline vs LSS
    每行：原图 | SegFormer 2D分割 | IPM BEV | LSS BEV
    """
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 SegFormer（用于 2D 分割）
    seg_processor = get_processor()
    seg_model = load_trained_model(
        f'{CKPT_DIR}/segformer_best.pth', device=device)

    # 加载 LSS 模型
    lss_model = LSSSegmentor(num_classes=NUM_CLASSES,
                             bev_size=BEV_SIZE,
                             bev_range=BEV_RANGE).to(device)
    lss_ckpt = f'{CKPT_DIR}/lss_best.pth'
    if os.path.exists(lss_ckpt):
        lss_model.load_state_dict(
            torch.load(lss_ckpt, map_location=device))
        print("LSS 权重加载成功")
    else:
        print("LSS 权重不存在，使用随机初始化（请先跑 train_bev.py）")
    lss_model.eval()

    samples = nusc.sample[:num_samples]
    fig, axes = plt.subplots(num_samples, 4,
                             figsize=(20, 5 * num_samples))

    for row, sample in enumerate(samples):
        cam_token = sample['data'][CAM_CHANNEL]
        cam_data  = nusc.get('sample_data', cam_token)
        cs = nusc.get('calibrated_sensor',
                      cam_data['calibrated_sensor_token'])
        intrinsic = np.array(cs['camera_intrinsic'], dtype=np.float32)

        img_path = DATA_ROOT + '/' + cam_data['filename']
        img_np   = np.array(Image.open(img_path).convert('RGB'))
        H, W     = img_np.shape[:2]

        # SegFormer 2D 推理
        encoding     = seg_processor(
            images=Image.fromarray(img_np), return_tensors='pt')
        pixel_values = encoding['pixel_values'].to(device)
        with torch.no_grad():
            outputs = seg_model(pixel_values=pixel_values)
        seg_mask = F.interpolate(
            outputs.logits, size=(H, W),
            mode='bilinear', align_corners=False
        ).argmax(dim=1).squeeze().cpu().numpy()

        # IPM BEV
        ipm_rgb = ipm_transform(img_np, intrinsic)
        ipm_sem = ipm_semantic(seg_mask, intrinsic)
        ipm_colored = COLORS[ipm_sem]

        # LSS BEV
        img_resized = np.array(
            Image.fromarray(img_np).resize((400, 224)))
        intr_scaled = intrinsic.copy()
        intr_scaled[0] *= 400 / W
        intr_scaled[1] *= 224 / H

        img_tensor = torch.from_numpy(
            img_resized.transpose(2, 0, 1)
        ).float().unsqueeze(0).to(device) / 255.0
        intr_tensor = torch.from_numpy(
            intr_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            lss_logits = lss_model(img_tensor, intr_tensor)
        lss_bev = lss_logits.argmax(dim=1).squeeze().cpu().numpy()
        lss_colored = COLORS[lss_bev]

        # 绘图
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title('Front Camera', fontsize=10)

        axes[row, 1].imshow(COLORS[seg_mask])
        axes[row, 1].set_title('SegFormer 2D', fontsize=10)

        axes[row, 2].imshow(ipm_colored)
        axes[row, 2].set_title('IPM BEV (baseline)', fontsize=10)

        axes[row, 3].imshow(lss_colored)
        axes[row, 3].set_title('LSS BEV', fontsize=10)

        for ax in axes[row]:
            ax.axis('off')

    # 图例
    patches = [mpatches.Patch(color=np.array(SEG_COLORS[i]) / 255,
                               label=ID2LABEL[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center',
               ncol=NUM_CLASSES, fontsize=10)

    plt.suptitle('IPM (baseline) vs LSS BEV Comparison', fontsize=14)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = f'{OUTPUT_DIR}/ipm_vs_lss.png'
    plt.savefig(save_path, dpi=120)
    print(f"对比图保存至：{save_path}")
    plt.show()


def visualize_ipm_only(nusc, num_samples=4):
    """
    只跑 IPM，不需要 LSS 权重
    用于训练 LSS 之前先看 IPM baseline 效果
    每行：原图 | SegFormer 2D | IPM RGB BEV | IPM 语义 BEV
    """
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg_processor = get_processor()
    seg_model = load_trained_model(
        f'{CKPT_DIR}/segformer_best.pth', device=device)

    samples = nusc.sample[:num_samples]
    fig, axes = plt.subplots(num_samples, 4,
                             figsize=(20, 5 * num_samples))

    for row, sample in enumerate(samples):
        cam_token = sample['data'][CAM_CHANNEL]
        cam_data  = nusc.get('sample_data', cam_token)
        cs = nusc.get('calibrated_sensor',
                      cam_data['calibrated_sensor_token'])
        intrinsic = np.array(cs['camera_intrinsic'], dtype=np.float32)

        img_np = np.array(
            Image.open(DATA_ROOT + '/' + cam_data['filename']).convert('RGB'))
        H, W   = img_np.shape[:2]

        encoding     = seg_processor(
            images=Image.fromarray(img_np), return_tensors='pt')
        pixel_values = encoding['pixel_values'].to(device)
        with torch.no_grad():
            outputs = seg_model(pixel_values=pixel_values)
        seg_mask = F.interpolate(
            outputs.logits, size=(H, W),
            mode='bilinear', align_corners=False
        ).argmax(dim=1).squeeze().cpu().numpy()

        ipm_rgb     = ipm_transform(img_np, intrinsic)
        ipm_sem     = ipm_semantic(seg_mask, intrinsic)
        ipm_colored = COLORS[ipm_sem]

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title('Front Camera', fontsize=10)
        axes[row, 1].imshow(COLORS[seg_mask])
        axes[row, 1].set_title('SegFormer 2D', fontsize=10)
        axes[row, 2].imshow(ipm_rgb)
        axes[row, 2].set_title('IPM RGB BEV', fontsize=10)
        axes[row, 3].imshow(ipm_colored)
        axes[row, 3].set_title('IPM Semantic BEV', fontsize=10)
        for ax in axes[row]:
            ax.axis('off')

    patches = [mpatches.Patch(color=np.array(SEG_COLORS[i]) / 255,
                               label=ID2LABEL[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center',
               ncol=NUM_CLASSES, fontsize=10)

    plt.suptitle('IPM BEV Baseline', fontsize=14)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = f'{OUTPUT_DIR}/ipm_baseline.png'
    plt.savefig(save_path, dpi=120)
    print(f"IPM baseline 图保存至：{save_path}")
    plt.show()


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=DATA_ROOT, verbose=False)
    # 先跑 IPM baseline，LSS 训练完再跑对比图
    visualize_ipm_only(nusc, num_samples=4)