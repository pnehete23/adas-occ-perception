# utils/visualize_depth.py
# 可视化 LiDAR sparse depth projection，证明 depth supervision 有真实信号
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, OUTPUT_DIR, CAM_CHANNEL
from utils.depth_utils import lidar_to_sparse_depth, depth_to_bin_label


def visualize_sparse_depth(nusc, sample_token,
                            cam_channel=CAM_CHANNEL,
                            save_path=None):
    """
    可视化单个 sample 的 sparse depth projection：
    左：原始图像
    中：LiDAR 点云投影到图像（颜色=深度）
    右：depth bin label（用于监督 DepthNet）
    """
    from PIL import Image as PILImage

    sample   = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)

    img = np.array(PILImage.open(
        os.path.join(DATA_ROOT, cam_data['filename'])).convert('RGB'))
    H, W = img.shape[:2]

    # 生成 sparse depth map
    depth_map   = lidar_to_sparse_depth(
        nusc, sample_token, cam_channel, (H, W))
    depth_label = depth_to_bin_label(depth_map)

    # 统计有深度值的像素数
    valid_pixels = (depth_map > 0).sum()
    total_pixels = H * W
    density = valid_pixels / total_pixels * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 左：原始图像
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=11)
    axes[0].axis('off')

    # 中：LiDAR 投影（颜色=深度值）
    axes[1].imshow(img, alpha=0.6)
    depth_vis = np.ma.masked_where(depth_map == 0, depth_map)
    # 用散点图代替 imshow，点更大更可见
    valid_mask = depth_map > 0
    ys, xs = np.where(valid_mask)
    depths  = depth_map[valid_mask]
    axes[1].imshow(img, alpha=0.7)
    sc = axes[1].scatter(xs, ys, c=depths, cmap='plasma',
                         vmin=1.0, vmax=45.0, s=1, alpha=0.9)
    plt.colorbar(sc, ax=axes[1], label='Depth (m)', fraction=0.046)
    axes[1].set_title(
        f'LiDAR Sparse Depth Projection\n'
        f'Valid pixels: {valid_pixels} / {total_pixels} '
        f'({density:.2f}%)', fontsize=10)
    axes[1].axis('off')

    # 右：depth bin label（监督信号）
    axes[2].imshow(img, alpha=0.6)
    label_vis = np.ma.masked_where(depth_label == 0, depth_label)
    valid_mask2 = depth_label > 0
    ys2, xs2 = np.where(valid_mask2)
    bins2 = depth_label[valid_mask2]
    axes[2].imshow(img, alpha=0.7)
    sc2 = axes[2].scatter(xs2, ys2, c=bins2, cmap='jet',
                          vmin=1, vmax=64, s=1, alpha=0.9)
    plt.colorbar(sc2, ax=axes[2],
                 label='Depth Bin Index (1-64)', fraction=0.046)
    axes[2].set_title(
        f'Depth Bin Labels for DepthNet Supervision\n'
        f'(64 bins, 1m to 45m)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(
        f'LiDAR Sparse Depth Supervision — {cam_channel}',
        fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()


def visualize_multicam_depth(nusc, sample_token, save_path=None):
    """
    可视化6路相机的 sparse depth projection
    """
    from PIL import Image as PILImage

    cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, cam in zip(axes.flatten(), cams):
        sample   = nusc.get('sample', sample_token)
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        img = np.array(PILImage.open(
            os.path.join(DATA_ROOT,
                         cam_data['filename'])).convert('RGB'))
        H, W = img.shape[:2]

        depth_map = lidar_to_sparse_depth(
            nusc, sample_token, cam, (H, W))
        valid = (depth_map > 0).sum()

        ax.imshow(img, alpha=0.7)
        valid_mask = depth_map > 0
        ys, xs = np.where(valid_mask)
        depths  = depth_map[valid_mask]
        ax.scatter(xs, ys, c=depths, cmap='plasma',
                   vmin=1.0, vmax=45.0, s=1, alpha=0.9)
        ax.set_title(
            f"{cam.replace('CAM_', '')}\n"
            f"valid pts: {valid}", fontsize=9)
        ax.axis('off')

    plt.suptitle(
        'LiDAR Sparse Depth Projection — 6 Cameras',
        fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version='v1.0-mini',
                    dataroot=DATA_ROOT, verbose=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 从不同 scene 各取第一个 sample
    selected_samples = []
    for scene in nusc.scene[:3]:
        first_sample_token = scene['first_sample_token']
        selected_samples.append(nusc.get('sample', first_sample_token))

    for i, sample in enumerate(selected_samples):
        visualize_sparse_depth(
            nusc, sample['token'],
            save_path=f'{OUTPUT_DIR}/sparse_depth_{i}.png')

    visualize_multicam_depth(
        nusc, selected_samples[0]['token'],
        save_path=f'{OUTPUT_DIR}/sparse_depth_6cam.png')

    print("Sparse depth 可视化完成！")