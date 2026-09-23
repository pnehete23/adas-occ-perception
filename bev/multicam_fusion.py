# bev/multicam_fusion.py
# 6-camera surround view BEV fusion demo
# 每个相机独立做 Lift，然后在 BEV 平面上 Splat 融合
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_ROOT, OUTPUT_DIR, BEV_SIZE, BEV_RANGE,
                    NUM_CLASSES, SEG_COLORS, ID2LABEL, CKPT_DIR)

CAM_CHANNELS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT'
]


def get_cam_params(nusc, sample_token, cam_channel, image_size=(224, 400)):
    """
    获取单个相机的图像、内参、外参
    Returns:
        img_tensor:  (3, H, W)
        intrinsic:   (3, 3)
        extrinsic:   (4, 4) camera-to-ego
    """
    from pyquaternion import Quaternion
    from PIL import Image as PILImage

    sample   = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs = nusc.get('calibrated_sensor',
                  cam_data['calibrated_sensor_token'])

    # 图像
    img = PILImage.open(
        os.path.join(DATA_ROOT, cam_data['filename'])).convert('RGB')
    W0, H0 = img.size
    img = img.resize((image_size[1], image_size[0]))
    img_tensor = torch.from_numpy(
        np.array(img).transpose(2, 0, 1)).float() / 255.0

    # 内参（随图像缩放调整）
    intrinsic = np.array(cs['camera_intrinsic'], dtype=np.float32)
    intrinsic[0] *= image_size[1] / W0
    intrinsic[1] *= image_size[0] / H0

    # 外参：camera-to-ego (4x4)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
    extrinsic[:3,  3] = np.array(cs['translation'])

    return img_tensor, intrinsic, extrinsic


def multicam_bev_fusion(nusc, sample_token, lss_model,
                         device, image_size=(224, 400)):
    """
    6路相机 BEV 特征融合
    每个相机独立提取特征 + Lift 到3D，然后在 BEV 平面累加

    Returns:
        fused_bev:   (C, bev_size, bev_size) 融合后的 BEV 特征
        cam_bevs:    dict {cam: (C, bev_size, bev_size)} 各相机单独的 BEV
    """
    lss_model.eval()
    cam_bevs = {}
    fused_bev = None

    with torch.no_grad():
        for cam in CAM_CHANNELS:
            try:
                img_tensor, intrinsic, extrinsic = get_cam_params(
                    nusc, sample_token, cam, image_size)

                img_t   = img_tensor.unsqueeze(0).to(device)
                intr_t  = torch.from_numpy(
                    intrinsic).unsqueeze(0).to(device)
                extr_t  = torch.from_numpy(
                    extrinsic).unsqueeze(0).to(device)

                # 只做 encoder + LSS lift/splat，不做 BEV decoder
                features = lss_model.encoder(img_t)
                depth_p  = lss_model.lss.depth_net(features)
                context  = lss_model.lss.context_net(features)
                points, weighted = lss_model.lss.lift(
                    context, depth_p, intr_t)

                # camera → ego frame
                ones     = torch.ones(*points.shape[:2], 1,
                                      device=device)
                points_h = torch.cat([points, ones], dim=-1)
                points   = torch.bmm(
                    points_h,
                    extr_t.transpose(1, 2))[:, :, :3]

                bev = lss_model.lss.splat(
                    points, weighted).squeeze(0)  # (C, S, S)

                cam_bevs[cam] = bev.cpu()

                if fused_bev is None:
                    fused_bev = bev
                else:
                    fused_bev = fused_bev + bev  # 累加融合

            except Exception as e:
                print(f"  {cam} 跳过: {e}")
                continue

    return fused_bev, cam_bevs


def visualize_multicam_fusion(nusc, sample_token,
                               lss_model, device,
                               save_path=None):
    """
    可视化6路相机 + 各相机 BEV + 融合 BEV
    """
    from PIL import Image as PILImage

    print(f"处理 sample: {sample_token[:8]}...")
    fused_bev, cam_bevs = multicam_bev_fusion(
        nusc, sample_token, lss_model, device)

    sample = nusc.get('sample', sample_token)

    # 图1：6路相机原图
    fig1, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, cam in zip(axes.flatten(), CAM_CHANNELS):
        cam_token = sample['data'][cam]
        cam_data  = nusc.get('sample_data', cam_token)
        img = plt.imread(
            os.path.join(DATA_ROOT, cam_data['filename']))
        ax.imshow(img)
        ax.set_title(cam.replace('CAM_', ''), fontsize=10)
        ax.axis('off')
    plt.suptitle('6-Camera Surround View', fontsize=13)
    plt.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_6cam.png')
        plt.savefig(p, dpi=120)
        print(f"保存至：{p}")
    plt.show()

    # 图2：各相机单独 BEV
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, cam in zip(axes.flatten(), CAM_CHANNELS):
        if cam in cam_bevs:
            bev = cam_bevs[cam]
            # 取最大激活通道做可视化
            bev_vis = bev.max(dim=0)[0].numpy()
            bev_vis = (bev_vis - bev_vis.min()) / \
                      (bev_vis.max() - bev_vis.min() + 1e-6)
            ax.imshow(bev_vis, cmap='hot', origin='lower')
            ax.set_title(cam.replace('CAM_', ''), fontsize=10)
        else:
            ax.set_title(f"{cam.replace('CAM_', '')} (skip)",
                         fontsize=10)
        ax.axis('off')
    plt.suptitle('Per-Camera BEV Feature Map', fontsize=13)
    plt.tight_layout()
    if save_path:
        p = save_path.replace('.png', '_percam_bev.png')
        plt.savefig(p, dpi=120)
        print(f"保存至：{p}")
    plt.show()

    # 图3：融合 BEV
    if fused_bev is not None:
        fused_vis = fused_bev.cpu().max(dim=0)[0].numpy()
        fused_vis = (fused_vis - fused_vis.min()) / \
                    (fused_vis.max() - fused_vis.min() + 1e-6)

        fig3, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(fused_vis, cmap='hot', origin='lower',
                       extent=[-BEV_RANGE/2, BEV_RANGE/2,
                                0, BEV_RANGE])
        plt.colorbar(im, ax=ax, label='Feature Activation')
        ax.set_xlabel('Y / Lateral (m)  left | right')
        ax.set_ylabel('X / Forward (m)')
        ax.set_title('6-Camera Fused BEV Feature Map\n'
                     '(ego-frame, surround view)', fontsize=12)
        ax.plot(0, 0, 'b^', markersize=12, label='Ego vehicle')
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"保存至：{save_path}")
        plt.show()

    return fused_bev, cam_bevs


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    from bev.lss import LSSSegmentor

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print(f"使用设备: {device}")

    nusc = NuScenes(version='v1.0-mini',
                    dataroot=DATA_ROOT, verbose=False)

    # 加载训练好的 LSS 模型
    model = LSSSegmentor(num_classes=NUM_CLASSES,
                         bev_size=BEV_SIZE,
                         bev_range=BEV_RANGE).to(device)
    ckpt = f'{CKPT_DIR}/lss_best.pth'
    if os.path.exists(ckpt):
        model.load_state_dict(
            torch.load(ckpt, map_location=device))
        print("LSS 权重加载成功")
    else:
        print("使用随机初始化权重（仅验证 pipeline）")
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected_samples = [nusc.get('sample', scene['first_sample_token'])
                        for scene in nusc.scene[:3]]
    for i, sample in enumerate(selected_samples):
        visualize_multicam_fusion(
            nusc, sample['token'], model, device,
            save_path=f'{OUTPUT_DIR}/multicam_fusion_{i}.png'
        )

    print("\n6-camera fusion demo 完成！")