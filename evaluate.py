# evaluate.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (CKPT_DIR, OUTPUT_DIR, NUM_CLASSES,
                    ID2LABEL, IMG_DIR, MASK_DIR)

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

segformer_mod = load_module('segformer',
    f'{DRIVE_ROOT}/models/segformer.py')
dataset_mod   = load_module('nuscenes_seg',
    f'{DRIVE_ROOT}/datasets/nuscenes_seg.py')

get_processor      = segformer_mod.get_processor
load_trained_model = segformer_mod.load_trained_model
NuScenesSegDataset = dataset_mod.NuScenesSegDataset


def compute_per_class_iou(preds, labels, num_classes=NUM_CLASSES):
    ious = {}
    preds  = preds.flatten()
    labels = labels.flatten()
    for cls in range(num_classes):
        pred_cls  = preds  == cls
        label_cls = labels == cls
        inter = (pred_cls & label_cls).sum()
        union = (pred_cls | label_cls).sum()
        if union > 0:
            ious[cls] = inter / union
        else:
            ious[cls] = None  # 该类别不存在
    return ious


def evaluate(ckpt_path, num_samples=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    processor = get_processor()
    model     = load_trained_model(ckpt_path, device=str(device))
    model.eval()

    val_dataset = NuScenesSegDataset(processor, split='val')
    val_loader  = DataLoader(val_dataset, batch_size=1,
                             shuffle=False, num_workers=2)

    # 累计混淆矩阵
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if num_samples and i >= num_samples:
                break
            pixel_values = batch['pixel_values'].to(device)
            labels       = batch['labels']

            outputs = model(pixel_values=pixel_values)
            upsampled = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode='bilinear', align_corners=False
            )
            preds = upsampled.argmax(dim=1).squeeze().cpu().numpy()
            labels_np = labels.squeeze().numpy()

            all_preds.append(preds.flatten())
            all_labels.append(labels_np.flatten())

            # 累计混淆矩阵
            for p, l in zip(preds.flatten(), labels_np.flatten()):
                conf_matrix[int(l), int(p)] += 1

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ── Per-class IoU ─────────────────────────────────
    print("\n── Per-Class IoU ──────────────────────────")
    class_ious = []
    for cls in range(NUM_CLASSES):
        pred_cls  = all_preds  == cls
        label_cls = all_labels == cls
        inter = (pred_cls & label_cls).sum()
        union = (pred_cls | label_cls).sum()
        if union > 0:
            iou = inter / union
            class_ious.append(iou)
            print(f"  {ID2LABEL[cls]:15s}: {iou:.4f}")
        else:
            class_ious.append(0.0)
            print(f"  {ID2LABEL[cls]:15s}: N/A (not present)")

    miou_present = np.mean([v for v in class_ious if v > 0])
    miou_all     = np.mean(class_ious)
    print(f"\n  mIoU (present classes only): {miou_present:.4f}")
    print(f"  mIoU (all eval classes):     {miou_all:.4f}")

    return conf_matrix, class_ious, miou_present, miou_all


def plot_confusion_matrix(conf_matrix, save_path=None):
    # 归一化
    conf_norm = conf_matrix.astype(float)
    row_sums  = conf_norm.sum(axis=1, keepdims=True)
    conf_norm = np.divide(conf_norm, row_sums,
                          where=row_sums != 0)

    labels = [ID2LABEL[i] for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：原始计数
    im0 = axes[0].imshow(conf_matrix, cmap='Blues')
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Ground Truth')
    axes[0].set_title('Confusion Matrix (counts)')
    plt.colorbar(im0, ax=axes[0])
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, str(conf_matrix[i, j]),
                         ha='center', va='center', fontsize=8,
                         color='white' if conf_matrix[i, j] > conf_matrix.max()/2
                         else 'black')

    # 右：归一化
    im1 = axes[1].imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Ground Truth')
    axes[1].set_title('Confusion Matrix (normalized)')
    plt.colorbar(im1, ax=axes[1])
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(j, i, f'{conf_norm[i, j]:.2f}',
                         ha='center', va='center', fontsize=8,
                         color='white' if conf_norm[i, j] > 0.5
                         else 'black')

    plt.suptitle('SegFormer Semantic Segmentation - Confusion Matrix',
                 fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()


def plot_per_class_iou(class_ious, miou, save_path=None):
    labels = [ID2LABEL[i] for i in range(NUM_CLASSES)]
    colors = ['#2196F3', '#F44336', '#9C27B0', '#4CAF50', '#FF9800']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, class_ious, color=colors, alpha=0.85)
    ax.axhline(y=miou, color='red', linestyle='--',
               label=f'mIoU: {miou:.4f}')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU - SegFormer on nuScenes Mini')
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, iou in zip(bars, class_ious):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{iou:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()


if __name__ == '__main__':
    ckpt_path = f'{CKPT_DIR}/segformer_best.pth'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conf_matrix, class_ious, miou_present, miou_all = evaluate(ckpt_path)

    plot_confusion_matrix(
        conf_matrix,
        save_path=f'{OUTPUT_DIR}/confusion_matrix.png'
    )
    plot_per_class_iou(
        class_ious, miou_present,
        save_path=f'{OUTPUT_DIR}/per_class_iou.png'
    )