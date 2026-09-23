# bev/ipm.py
# Inverse Perspective Mapping - BEV baseline
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)
from config import BEV_RANGE, BEV_SIZE, CAM_HEIGHT, OUTPUT_DIR


def ipm_transform(img: np.ndarray,
                  intrinsic: np.ndarray,
                  bev_range: float = BEV_RANGE,
                  bev_size:  int   = BEV_SIZE,
                  cam_height: float = CAM_HEIGHT) -> np.ndarray:
    """
    逆透视变换：把前视图投影到 BEV 鸟瞰图
    假设地面平坦，相机水平安装

    Args:
        img:        原始图像 (H, W, 3)
        intrinsic:  3x3 相机内参矩阵
        bev_range:  BEV 覆盖的前方距离（米）
        bev_size:   输出 BEV 图像分辨率
        cam_height: 相机离地高度（米）

    Returns:
        bev_img: BEV 图像 (bev_size, bev_size, 3)
    """
    H, W = img.shape[:2]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    bev_img = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    # 构建 BEV 网格，向量化加速
    bev_v, bev_u = np.meshgrid(
        np.arange(bev_size), np.arange(bev_size), indexing='ij'
    )
    # BEV 像素 → 真实世界坐标（米）
    x = (bev_u / bev_size - 0.5) * bev_range   # 横向（左负右正）
    z = (1 - bev_v / bev_size) * bev_range      # 纵向（前方距离）

    valid = z > 1.0  # 过滤太近的点

    # 真实坐标 → 图像像素坐标
    img_u = np.where(valid, (fx * x / z + cx).astype(int), -1)
    img_v = np.where(valid, (fy * cam_height / z + cy).astype(int), -1)

    # 只取在图像范围内的点
    in_bound = (img_u >= 0) & (img_u < W) & (img_v >= 0) & (img_v < H)
    bev_img[in_bound] = img[img_v[in_bound], img_u[in_bound]]

    return bev_img


def ipm_semantic(seg_mask: np.ndarray,
                 intrinsic: np.ndarray,
                 bev_range: float = BEV_RANGE,
                 bev_size:  int   = BEV_SIZE,
                 cam_height: float = CAM_HEIGHT) -> np.ndarray:
    """
    把语义分割 mask 投影到 BEV
    返回 BEV 语义 mask (bev_size, bev_size)
    """
    H, W = seg_mask.shape
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    bev_mask = np.zeros((bev_size, bev_size), dtype=np.uint8)

    bev_v, bev_u = np.meshgrid(
        np.arange(bev_size), np.arange(bev_size), indexing='ij'
    )
    x = (bev_u / bev_size - 0.5) * bev_range
    z = (1 - bev_v / bev_size) * bev_range
    valid = z > 1.0

    img_u = np.where(valid, (fx * x / z + cx).astype(int), -1)
    img_v = np.where(valid, (fy * cam_height / z + cy).astype(int), -1)

    in_bound = (img_u >= 0) & (img_u < W) & (img_v >= 0) & (img_v < H)
    bev_mask[in_bound] = seg_mask[img_v[in_bound], img_u[in_bound]]

    return bev_mask


def visualize_ipm(nusc, sample_token, model, processor,
                  device, save_path=None):
    """
    对一个 sample 做完整的 IPM 可视化：
    原图 | SegFormer分割 | RGB BEV | 语义BEV
    """
    import torch
    import torch.nn.functional as F
    from config import SEG_COLORS

    sample   = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    intrinsic = np.array(cs['camera_intrinsic'])

    from config import DATA_ROOT
    img = np.array(Image.open(DATA_ROOT + '/' + cam_data['filename']))

    # SegFormer 推理
    encoding     = processor(images=Image.fromarray(img), return_tensors='pt')
    pixel_values = encoding['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    upsampled = F.interpolate(
        outputs.logits, size=(img.shape[0], img.shape[1]),
        mode='bilinear', align_corners=False
    )
    seg_mask = upsampled.argmax(dim=1).squeeze().cpu().numpy()

    # IPM
    bev_rgb  = ipm_transform(img, intrinsic)
    bev_sem  = ipm_semantic(seg_mask, intrinsic)

    # 语义 BEV 着色
    colors   = np.array([SEG_COLORS[i] for i in range(len(SEG_COLORS))],
                        dtype=np.uint8)
    bev_colored = colors[bev_sem]

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img);           axes[0].set_title('Front Camera')
    axes[1].imshow(colors[seg_mask]); axes[1].set_title('SegFormer 2D')
    axes[2].imshow(bev_rgb);       axes[2].set_title('IPM RGB BEV')
    axes[3].imshow(bev_colored);   axes[3].set_title('IPM Semantic BEV')
    for ax in axes:
        ax.axis('off')

    plt.suptitle('IPM Inverse Perspective Mapping', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()