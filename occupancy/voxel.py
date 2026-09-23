# occupancy/voxel.py
# 从 LiDAR 点云生成 3D 体素占用网格
import numpy as np
import sys

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)
from config import BEV_RANGE, NUM_CLASSES, CAT_MAP


# 体素网格参数
VOXEL_X_RANGE = (-25.0, 25.0)   # 左右各25米
VOXEL_Y_RANGE = (-25.0, 25.0)   # 前后各25米
VOXEL_Z_RANGE = (-2.0,  4.0)    # 高度：地下2米到地上4米
VOXEL_SIZE    = 0.5              # 每个体素0.5米


def get_voxel_grid_shape():
    nx = int((VOXEL_X_RANGE[1] - VOXEL_X_RANGE[0]) / VOXEL_SIZE)
    ny = int((VOXEL_Y_RANGE[1] - VOXEL_Y_RANGE[0]) / VOXEL_SIZE)
    nz = int((VOXEL_Z_RANGE[1] - VOXEL_Z_RANGE[0]) / VOXEL_SIZE)
    return nx, ny, nz  # (100, 100, 12)


def lidar_to_voxel(nusc, sample_token):
    """
    把 LiDAR 点云转换为3D体素占用网格
    返回:
        voxel_grid: (nx, ny, nz) uint8，0=free, 1=occupied
        points_ego: (N, 3) ego 坐标系下的点云
    """
    from pyquaternion import Quaternion

    sample    = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data  = nusc.get('sample_data', lidar_token)

    # 加载点云
    from nuscenes.utils.data_classes import LidarPointCloud
    import os
    from config import DATA_ROOT
    pc = LidarPointCloud.from_file(
        os.path.join(DATA_ROOT, lidar_data['filename']))

    # LiDAR → ego 坐标系
    cs  = nusc.get('calibrated_sensor',
                   lidar_data['calibrated_sensor_token'])
    ego = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
    pc.translate(np.array(cs['translation']))

    points = pc.points[:3, :].T  # (N, 3) xyz

    # 过滤范围外的点
    mask = (
        (points[:, 0] >= VOXEL_X_RANGE[0]) &
        (points[:, 0] <  VOXEL_X_RANGE[1]) &
        (points[:, 1] >= VOXEL_Y_RANGE[0]) &
        (points[:, 1] <  VOXEL_Y_RANGE[1]) &
        (points[:, 2] >= VOXEL_Z_RANGE[0]) &
        (points[:, 2] <  VOXEL_Z_RANGE[1])
    )
    points = points[mask]

    # 点云坐标 → 体素索引
    nx, ny, nz = get_voxel_grid_shape()
    ix = ((points[:, 0] - VOXEL_X_RANGE[0]) / VOXEL_SIZE).astype(int)
    iy = ((points[:, 1] - VOXEL_Y_RANGE[0]) / VOXEL_SIZE).astype(int)
    iz = ((points[:, 2] - VOXEL_Z_RANGE[0]) / VOXEL_SIZE).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)

    voxel_grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    voxel_grid[ix, iy, iz] = 1

    return voxel_grid, points


def lidar_to_semantic_voxel(nusc, sample_token):
    from pyquaternion import Quaternion

    sample = nusc.get('sample', sample_token)
    nx, ny, nz = get_voxel_grid_shape()

    # Step 1: 用 LiDAR 点云确定哪些体素有点
    voxel_occupied, points = lidar_to_voxel(nusc, sample_token)

    # Step 2: 建立体素索引到点的映射
    from config import DATA_ROOT
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data  = nusc.get('sample_data', lidar_token)
    ego = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # 计算每个点的体素索引
    ix = ((points[:, 0] - VOXEL_X_RANGE[0]) / VOXEL_SIZE).astype(int)
    iy = ((points[:, 1] - VOXEL_Y_RANGE[0]) / VOXEL_SIZE).astype(int)
    iz = ((points[:, 2] - VOXEL_Z_RANGE[0]) / VOXEL_SIZE).astype(int)
    ix = np.clip(ix, 0, nx-1)
    iy = np.clip(iy, 0, ny-1)
    iz = np.clip(iz, 0, nz-1)

    # Step 3: 用3D box 给每个点打语义标签
    semantic_voxel = np.zeros((nx, ny, nz), dtype=np.uint8)

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

        cx, cy, cz = box.center
        lx, ly, lz = box.wlh[1], box.wlh[0], box.wlh[2]

        # 找出落在这个 box 内的 LiDAR 点
        in_box = (
            (points[:, 0] >= cx - lx/2) & (points[:, 0] <= cx + lx/2) &
            (points[:, 1] >= cy - ly/2) & (points[:, 1] <= cy + ly/2) &
            (points[:, 2] >= cz - lz/2) & (points[:, 2] <= cz + lz/2)
        )

        # 给这些点对应的体素打标签
        semantic_voxel[ix[in_box], iy[in_box], iz[in_box]] = label

    # Step 4: 没有语义标签的 occupied 体素标为 background（0）
    # 有语义的体素保持语义标签
    return semantic_voxel


def lidar_to_semantic_voxel_multiframe(nusc, sample_token,
                                        num_sweeps=5):
    """
    多帧点云聚合 + ego-motion 补偿
    num_sweeps: 往前聚合多少帧 sweep
    """
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud
    from config import DATA_ROOT
    import os

    sample   = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data  = nusc.get('sample_data', lidar_token)

    # 当前帧的 ego pose（作为参考坐标系）
    ego_ref = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_ref_rot = Quaternion(ego_ref['rotation'])
    ego_ref_trans = np.array(ego_ref['translation'])

    # 当前帧相机标定
    cs_ref = nusc.get('calibrated_sensor',
                      lidar_data['calibrated_sensor_token'])

    all_points = []

    # 收集多帧点云
    current_token = lidar_token
    for sweep_idx in range(num_sweeps):
        sd = nusc.get('sample_data', current_token)
        cs  = nusc.get('calibrated_sensor',
                       sd['calibrated_sensor_token'])
        ego = nusc.get('ego_pose', sd['ego_pose_token'])

        # 加载点云
        pc = LidarPointCloud.from_file(
            os.path.join(DATA_ROOT, sd['filename']))

        # 1. LiDAR → ego（当前帧）
        pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
        pc.translate(np.array(cs['translation']))

        # 2. ego（当前帧）→ 全局坐标
        pc.rotate(Quaternion(ego['rotation']).rotation_matrix)
        pc.translate(np.array(ego['translation']))

        # 3. 全局坐标 → ego（参考帧，即 keyframe）
        pc.translate(-ego_ref_trans)
        pc.rotate(ego_ref_rot.inverse.rotation_matrix)

        pts = pc.points[:3, :].T  # (N, 3)
        all_points.append(pts)

        # 往前一帧
        if sd['prev'] == '':
            break
        current_token = sd['prev']

    # 合并所有帧
    all_points = np.concatenate(all_points, axis=0)

    # 过滤范围
    mask = (
        (all_points[:, 0] >= VOXEL_X_RANGE[0]) &
        (all_points[:, 0] <  VOXEL_X_RANGE[1]) &
        (all_points[:, 1] >= VOXEL_Y_RANGE[0]) &
        (all_points[:, 1] <  VOXEL_Y_RANGE[1]) &
        (all_points[:, 2] >= VOXEL_Z_RANGE[0]) &
        (all_points[:, 2] <  VOXEL_Z_RANGE[1])
    )
    points = all_points[mask]

    # 体素化
    nx, ny, nz = get_voxel_grid_shape()
    ix = ((points[:, 0] - VOXEL_X_RANGE[0]) / VOXEL_SIZE).astype(int)
    iy = ((points[:, 1] - VOXEL_Y_RANGE[0]) / VOXEL_SIZE).astype(int)
    iz = ((points[:, 2] - VOXEL_Z_RANGE[0]) / VOXEL_SIZE).astype(int)
    ix = np.clip(ix, 0, nx-1)
    iy = np.clip(iy, 0, ny-1)
    iz = np.clip(iz, 0, nz-1)

    # 语义标签（用3D box）
    ego = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    semantic_voxel = np.zeros((nx, ny, nz), dtype=np.uint8)

    # 先把所有点标为 occupied（background）
    # 再用 box 覆盖语义类别
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

        cx, cy, cz = box.center
        lx, ly, lz = box.wlh[1], box.wlh[0], box.wlh[2]

        # 用完整 box 填充体素（给视觉体积感）
        x1 = int(np.clip((cx - lx/2 - VOXEL_X_RANGE[0]) / VOXEL_SIZE, 0, nx-1))
        x2 = int(np.clip((cx + lx/2 - VOXEL_X_RANGE[0]) / VOXEL_SIZE, 0, nx-1))
        y1 = int(np.clip((cy - ly/2 - VOXEL_Y_RANGE[0]) / VOXEL_SIZE, 0, ny-1))
        y2 = int(np.clip((cy + ly/2 - VOXEL_Y_RANGE[0]) / VOXEL_SIZE, 0, ny-1))
        z1 = int(np.clip((cz - lz/2 - VOXEL_Z_RANGE[0]) / VOXEL_SIZE, 0, nz-1))
        z2 = int(np.clip((cz + lz/2 - VOXEL_Z_RANGE[0]) / VOXEL_SIZE, 0, nz-1))
        if x2 >= x1 and y2 >= y1 and z2 >= z1:
            semantic_voxel[x1:x2+1, y1:y2+1, z1:z2+1] = label

    print(f"多帧聚合：{len(all_points)} 个点（{num_sweeps} 帧）")
    return semantic_voxel, points