# occupancy/visualize_occ.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import OUTPUT_DIR, SEG_COLORS, ID2LABEL, DATA_ROOT
from occupancy.voxel import (lidar_to_voxel, lidar_to_semantic_voxel,
                              get_voxel_grid_shape,
                              VOXEL_X_RANGE, VOXEL_Y_RANGE,
                              VOXEL_Z_RANGE, VOXEL_SIZE)


def visualize_lidar_bev(nusc, sample_token, save_path=None):
    """
    俯视图（BEV）可视化 LiDAR 点云
    """
    _, points = lidar_to_voxel(nusc, sample_token)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1],
               s=0.3, c='lime', alpha=0.5)
    ax.set_xlim(VOXEL_X_RANGE)
    ax.set_ylim(VOXEL_Y_RANGE)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('LiDAR Point Cloud - BEV')
    ax.set_facecolor('black')
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()


def visualize_semantic_voxel_bev(nusc, sample_token, save_path=None):
    """
    语义体素 BEV 俯视图（沿 Z 轴压平）
    """
    semantic_voxel = lidar_to_semantic_voxel(nusc, sample_token)

    # 沿高度轴取最大类别（忽略 free=0）
    bev_semantic = np.zeros(semantic_voxel.shape[:2], dtype=np.uint8)
    for z in range(semantic_voxel.shape[2]):
        layer = semantic_voxel[:, :, z]
        bev_semantic = np.where(layer > 0, layer, bev_semantic)

    # 着色
    colors = np.array([SEG_COLORS[i] for i in range(len(SEG_COLORS))],
                      dtype=np.uint8)
    bev_colored = colors[bev_semantic]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：语义 BEV
    axes[0].imshow(bev_colored.transpose(1, 0, 2),
                   origin='lower', extent=[
                       VOXEL_X_RANGE[0], VOXEL_X_RANGE[1],
                       VOXEL_Y_RANGE[0], VOXEL_Y_RANGE[1]])
    axes[0].set_title('Semantic Occupancy BEV (top-down)', fontsize=12)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')

    # 右：各层高度切片（显示4个高度层）
    nz = semantic_voxel.shape[2]
    z_indices = [0, nz//4, nz//2, nz-1]
    z_heights  = [VOXEL_Z_RANGE[0] + i * VOXEL_SIZE for i in z_indices]

    slice_grid = np.zeros((semantic_voxel.shape[0] * 2,
                           semantic_voxel.shape[1] * 2, 3), dtype=np.uint8)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    nx, ny = semantic_voxel.shape[:2]

    for (r, c), zi, zh in zip(positions, z_indices, z_heights):
        layer = semantic_voxel[:, :, zi]
        colored = colors[layer]
        slice_grid[r*nx:(r+1)*nx, c*ny:(c+1)*ny] = colored

    axes[1].imshow(slice_grid.transpose(1, 0, 2), origin='lower')
    axes[1].set_title('Height Slices: z={:.1f}, {:.1f}, {:.1f}, {:.1f}m'.format(
        *z_heights), fontsize=10)
    axes[1].axis('off')

    patches = [mpatches.Patch(color=np.array(SEG_COLORS[i])/255,
                               label=ID2LABEL[i])
               for i in range(len(ID2LABEL))]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=10)
    plt.suptitle('3D Semantic Occupancy Grid', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()


def visualize_3d_voxel(nusc, sample_token, save_path=None, max_points=20000):
    """
    3D 散点图可视化语义体素
    """
    semantic_voxel = lidar_to_semantic_voxel(nusc, sample_token)
    nx, ny, nz = get_voxel_grid_shape()

    # 收集所有非空体素的坐标和类别
    xs, ys, zs, labels = [], [], [], []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                label = semantic_voxel[ix, iy, iz]
                if label > 0:
                    xs.append(VOXEL_X_RANGE[0] + ix * VOXEL_SIZE)
                    ys.append(VOXEL_Y_RANGE[0] + iy * VOXEL_SIZE)
                    zs.append(VOXEL_Z_RANGE[0] + iz * VOXEL_SIZE)
                    labels.append(label)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    labels = np.array(labels)

    # 随机采样避免太多点
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), max_points, replace=False)
        xs, ys, zs, labels = xs[idx], ys[idx], zs[idx], labels[idx]

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    for label_id in range(1, len(ID2LABEL)):
        mask = labels == label_id
        if mask.sum() == 0:
            continue
        color = np.array(SEG_COLORS[label_id]) / 255
        ax.scatter(xs[mask], ys[mask], zs[mask],
                   c=[color], s=20, alpha=0.7,
                   label=ID2LABEL[label_id])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Semantic Occupancy Grid', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('#1a1a1a')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()


def run_occupancy_visualization(nusc, num_samples=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected_samples = [nusc.get('sample', scene['first_sample_token'])
                        for scene in nusc.scene[:num_samples]]
    for i, sample in enumerate(selected_samples):
        token = sample['token']
        print(f"\nSample {i+1}/{num_samples}: {token[:8]}...")

        # LiDAR BEV（单帧）
        visualize_lidar_bev(
            nusc, token,
            save_path=f'{OUTPUT_DIR}/lidar_bev_{i}.png')

        # 多帧聚合 Occupancy
        from occupancy.voxel import lidar_to_semantic_voxel_multiframe
        semantic_voxel, points = lidar_to_semantic_voxel_multiframe(
            nusc, token, num_sweeps=5)

        # 3D 可视化（直接用聚合后的点和语义体素）
        _visualize_multiframe_3d(
            semantic_voxel, points,
            save_path=f'{OUTPUT_DIR}/semantic_occ_3d_{i}.png')

    print("\nOccupancy 可视化完成！")


def _visualize_multiframe_3d(semantic_voxel, points,
                              save_path=None, max_points=30000):
    from config import SEG_COLORS, ID2LABEL
    from occupancy.voxel import (VOXEL_X_RANGE, VOXEL_Y_RANGE,
                                  VOXEL_Z_RANGE, VOXEL_SIZE,
                                  get_voxel_grid_shape)

    nx, ny, nz = get_voxel_grid_shape()

    xs, ys, zs, labels = [], [], [], []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                label = semantic_voxel[ix, iy, iz]
                if label > 0:
                    xs.append(VOXEL_X_RANGE[0] + ix * VOXEL_SIZE)
                    ys.append(VOXEL_Y_RANGE[0] + iy * VOXEL_SIZE)
                    zs.append(VOXEL_Z_RANGE[0] + iz * VOXEL_SIZE)
                    labels.append(label)

    # 同时显示背景点云（灰色）
    bg_pts = points
    if len(bg_pts) > max_points:
        idx = np.random.choice(len(bg_pts), max_points, replace=False)
        bg_pts = bg_pts[idx]

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    labels = np.array(labels)

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # 背景点云
    ax.scatter(bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
               c='#404040', s=0.1, alpha=0.2)

    # 语义体素
    for label_id in range(1, len(ID2LABEL)):
        mask = labels == label_id
        if mask.sum() == 0:
            continue
        color = np.array(SEG_COLORS[label_id]) / 255
        ax.scatter(xs[mask], ys[mask], zs[mask],
                   c=[color], s=200, alpha=0.9,
                   label=ID2LABEL[label_id],
                   edgecolors='none')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(VOXEL_X_RANGE)
    ax.set_ylim(VOXEL_Y_RANGE)
    ax.set_zlim(VOXEL_Z_RANGE)
    ax.set_title('3D Semantic Occupancy (5-frame aggregation)', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('#1a1a1a')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"保存至：{save_path}")
    plt.show()


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=DATA_ROOT, verbose=False)
    run_occupancy_visualization(nusc, num_samples=3)