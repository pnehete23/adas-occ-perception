# bev/lss.py
# Lift-Splat-Shoot: 把图像特征提升到 3D 再拍平到 BEV
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)
from config import BEV_RANGE, BEV_SIZE


class DepthNet(nn.Module):
    """
    预测每个像素的深度分布（离散化为 D 个 bin）
    输入: 图像特征 (B, C, H, W)
    输出: 深度概率分布 (B, D, H, W)
    """
    def __init__(self, in_channels=256, depth_bins=64):
        super().__init__()
        self.depth_bins = depth_bins
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, depth_bins, 1),
        )

    def forward(self, x):
        return self.net(x).softmax(dim=1)  # (B, D, H, W)


class ContextNet(nn.Module):
    """
    提取每个像素的语义特征
    输入: 图像特征 (B, C, H, W)
    输出: 上下文特征 (B, C_out, H, W)
    """
    def __init__(self, in_channels=256, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)  # (B, C_out, H, W)


class LSSTransform(nn.Module):
    """
    LSS 核心模块：Lift-Splat-Shoot
    
    Lift:   为每个像素生成 D 个深度候选点，
            结合语义特征形成点云 (B, D*H*W, C)
    Splat:  把点云体素化，投影到 BEV 平面
    Shoot:  BEV 特征用于下游任务（分割/检测）
    """
    def __init__(self,
                 image_size=(32, 56),   # 下采样后的特征图尺寸
                 depth_bins=64,
                 depth_min=1.0,
                 depth_max=45.0,
                 bev_size=BEV_SIZE,
                 bev_range=BEV_RANGE,
                 feature_channels=64):
        super().__init__()
        self.image_size      = image_size
        self.depth_bins      = depth_bins
        self.depth_min       = depth_min
        self.depth_max       = depth_max
        self.bev_size        = bev_size
        self.bev_range       = bev_range
        self.feature_channels = feature_channels

        # 深度 bin 中心值
        self.register_buffer('depth_values',
            torch.linspace(depth_min, depth_max, depth_bins))

        self.depth_net   = DepthNet(in_channels=256,
                                    depth_bins=depth_bins)
        self.context_net = ContextNet(in_channels=256,
                                      out_channels=feature_channels)

    def lift(self, features, depth_probs, intrinsic):
        """
        Lift: 图像特征 → 3D 点云
        features:    (B, C, H, W)
        depth_probs: (B, D, H, W)
        intrinsic:   (B, 3, 3)
        返回:
            points:   (B, D*H*W, 3)  3D坐标
            feats:    (B, D*H*W, C)  对应特征
        """
        B, C, H, W = features.shape
        D = self.depth_bins
        device = features.device

        # 构建像素坐标网格
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        uu, vv = torch.meshgrid(u, v, indexing='xy')  # (H, W)

        # 逆投影：像素 → 相机坐标（单位深度）
        fx = intrinsic[:, 0, 0].view(B, 1, 1)  # (B,1,1)
        fy = intrinsic[:, 1, 1].view(B, 1, 1)
        cx = intrinsic[:, 0, 2].view(B, 1, 1)
        cy = intrinsic[:, 1, 2].view(B, 1, 1)

        x_norm = (uu.unsqueeze(0) - cx) / fx  # (B, H, W)
        y_norm = (vv.unsqueeze(0) - cy) / fy

        # 扩展到 D 个深度 bin
        depths = self.depth_values.view(1, D, 1, 1)  # (1,D,1,1)
        x_3d = (x_norm.unsqueeze(1) * depths)        # (B,D,H,W)
        y_3d = (y_norm.unsqueeze(1) * depths)
        z_3d = depths.expand(B, D, H, W)

        # 拼接成点云
        points = torch.stack([x_3d, y_3d, z_3d], dim=-1)  # (B,D,H,W,3)
        points = points.view(B, D * H * W, 3)

        # 加权特征：depth_prob * context_feat
        # depth_probs: (B,D,H,W) → (B,D,H,W,1)
        # features:    (B,C,H,W) → (B,1,H,W,C)
        dp = depth_probs.unsqueeze(-1)              # (B,D,H,W,1)
        ft = features.permute(0, 2, 3, 1).unsqueeze(1)  # (B,1,H,W,C)
        weighted = (dp * ft).reshape(B, D * H * W, C)  # (B,D*H*W,C)

        return points, weighted

    def splat(self, points, feats):
        B, N, C = feats.shape
        device  = feats.device
        S = self.bev_size

        # nuScenes ego frame: x=forward, y=left, z=up
        # BEV 图约定：图像上方=前方(x)，图像右方=右方(-y)
        px = ((points[:, :, 1] * -1 + self.bev_range / 2) / self.bev_range * S)  # y轴（左右）
        pz = ((1 - points[:, :, 0] / self.bev_range) * S)                         # x轴（前后）

        # 归一化到 [-1, 1] 用于 grid_sample
        grid_x = (px / S) * 2 - 1  # (B, N)
        grid_z = (pz / S) * 2 - 1  # (B, N)

        # 只取有效点（在 BEV 范围内且在相机前方）
        valid = (grid_x > -1) & (grid_x < 1) & \
                (grid_z > -1) & (grid_z < 1) & \
                (points[:, :, 2] > 0)

        # 用 scatter 累加到 BEV 网格
        bev_feat = torch.zeros(B, C, S, S, device=device)
        for b in range(B):
            v = valid[b]
            if v.sum() == 0:
                continue
            ix = px[b][v].long().clamp(0, S - 1)
            iz = pz[b][v].long().clamp(0, S - 1)
            f  = feats[b][v]  # (N_valid, C)
            idx = iz * S + ix  # 展平索引
            bev_flat = bev_feat[b].view(C, S * S)
            bev_flat.scatter_add_(
                1,
                idx.unsqueeze(0).expand(C, -1),
                f.T
            )
        return bev_feat

    def forward(self, features, intrinsic, extrinsic=None):
        """
        features:   (B, 256, H, W)  backbone 输出特征
        intrinsic:  (B, 3, 3)       相机内参
        extrinsic:  (B, 4, 4)       camera-to-ego 外参（可选）
        返回:
            bev_feat: (B, C, bev_size, bev_size)
        """
        depth_probs = self.depth_net(features)
        context     = self.context_net(features)
        points, weighted_feats = self.lift(context, depth_probs, intrinsic)

        # 如果提供外参，把点从 camera frame 变换到 ego frame
        if extrinsic is not None:
            ones = torch.ones(*points.shape[:2], 1, device=points.device)
            points_h = torch.cat([points, ones], dim=-1)  # (B, N, 4)
            points = torch.bmm(points_h, extrinsic.transpose(1, 2))[:, :, :3]

        bev_feat = self.splat(points, weighted_feats)
        return bev_feat


class LSSSegmentor(nn.Module):
    """
    完整的 LSS 语义分割网络：
    Encoder（EfficientNet风格简化版）→ LSS → BEV Decoder → 语义输出
    """
    def __init__(self, num_classes=5,
                 bev_size=BEV_SIZE,
                 bev_range=BEV_RANGE):
        super().__init__()

        # 简化版 Encoder：把图像压缩到特征图
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   32,  3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.lss = LSSTransform(
            depth_bins=64,
            bev_size=bev_size,
            bev_range=bev_range,
            feature_channels=64,
        )

        # BEV Decoder：64通道特征 → num_classes
        self.bev_decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, images, intrinsics, extrinsics=None,
                return_depth=False):
        features    = self.encoder(images)
        depth_probs = self.lss.depth_net(features)
        context     = self.lss.context_net(features)
        points, weighted_feats = self.lss.lift(
            context, depth_probs, intrinsic=intrinsics)
        if extrinsics is not None:
            ones     = torch.ones(*points.shape[:2], 1,
                                  device=points.device)
            points_h = torch.cat([points, ones], dim=-1)
            points   = torch.bmm(
                points_h, extrinsics.transpose(1, 2))[:, :, :3]
        bev_feat   = self.lss.splat(points, weighted_feats)
        bev_logits = self.bev_decoder(bev_feat)
        if return_depth:
            return bev_logits, depth_probs
        return bev_logits