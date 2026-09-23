# utils/depth_utils.py
# LiDAR 点云投影到图像平面，生成 sparse depth label
import numpy as np
import torch
import sys

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)


def lidar_to_sparse_depth(nusc, sample_token, cam_channel,
                           image_size, depth_min=1.0, depth_max=45.0):
    """
    把 LiDAR 点云投影到相机图像平面，生成稀疏深度图

    Args:
        nusc:         NuScenes 实例
        sample_token: sample token
        cam_channel:  相机通道名
        image_size:   (H, W) 目标图像尺寸
        depth_min:    最小深度（米）
        depth_max:    最大深度（米）

    Returns:
        depth_map: (H, W) float32，0 表示无深度值
    """
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import LidarPointCloud
    from config import DATA_ROOT
    import os

    sample    = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    lid_token = sample['data']['LIDAR_TOP']

    cam_data = nusc.get('sample_data', cam_token)
    lid_data = nusc.get('sample_data', lid_token)

    cam_cs  = nusc.get('calibrated_sensor',
                       cam_data['calibrated_sensor_token'])
    cam_ego = nusc.get('ego_pose', cam_data['ego_pose_token'])
    lid_cs  = nusc.get('calibrated_sensor',
                       lid_data['calibrated_sensor_token'])
    lid_ego = nusc.get('ego_pose', lid_data['ego_pose_token'])

    # 加载 LiDAR 点云
    pc = LidarPointCloud.from_file(
        os.path.join(DATA_ROOT, lid_data['filename']))

    # LiDAR → ego（LiDAR 时刻）
    pc.rotate(Quaternion(lid_cs['rotation']).rotation_matrix)
    pc.translate(np.array(lid_cs['translation']))

    # ego（LiDAR）→ 全局
    pc.rotate(Quaternion(lid_ego['rotation']).rotation_matrix)
    pc.translate(np.array(lid_ego['translation']))

    # 全局 → ego（相机时刻）
    pc.translate(-np.array(cam_ego['translation']))
    pc.rotate(Quaternion(cam_ego['rotation']).inverse.rotation_matrix)

    # ego（相机）→ camera frame
    pc.translate(-np.array(cam_cs['translation']))
    pc.rotate(Quaternion(cam_cs['rotation']).inverse.rotation_matrix)

    points = pc.points[:3, :].T  # (N, 3) in camera frame

    # 只保留相机前方的点
    valid = points[:, 2] > depth_min
    points = points[valid]

    # 投影到图像平面
    intrinsic = np.array(cam_cs['camera_intrinsic'])
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = (fx * points[:, 0] / points[:, 2] + cx).astype(int)
    v = (fy * points[:, 1] / points[:, 2] + cy).astype(int)
    d = points[:, 2]

    # 生成深度图
    H, W = image_size
    depth_map = np.zeros((H, W), dtype=np.float32)

    in_bound = (u >= 0) & (u < W) & (v >= 0) & (v < H) & \
               (d >= depth_min) & (d <= depth_max)

    u, v, d = u[in_bound], v[in_bound], d[in_bound]

    # 深度值归一化到 [0, 1]（按 depth bin 离散化）
    depth_map[v, u] = d

    return depth_map


def depth_to_bin_label(depth_map, depth_min=1.0,
                        depth_max=45.0, num_bins=64):
    """
    把连续深度图转成深度 bin 索引（用于监督 DepthNet）

    Returns:
        bin_label: (H, W) int64，0 表示无深度，1~num_bins 表示对应 bin
    """
    H, W = depth_map.shape
    bin_label = np.zeros((H, W), dtype=np.int64)

    valid = depth_map > 0
    d     = depth_map[valid]

    # 线性 bin 划分
    bins = np.linspace(depth_min, depth_max, num_bins + 1)
    idx  = np.digitize(d, bins) - 1
    idx  = np.clip(idx, 0, num_bins - 1)

    bin_label[valid] = idx + 1  # 0 留给无深度
    return bin_label


def resize_depth(depth_map, target_size):
    """
    把深度图 resize 到模型特征图尺寸（最近邻插值保持深度值）
    target_size: (H, W)
    """
    from PIL import Image as PILImage
    import numpy as np
    d = PILImage.fromarray(depth_map.astype(np.int32))
    d = d.resize((target_size[1], target_size[0]),
                 PILImage.NEAREST)
    return np.array(d)