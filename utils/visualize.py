# utils/visualize.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
sys.path.insert(0, '/content/drive/MyDrive/adas_occ_project')
from config import SEG_COLORS, ID2LABEL

COLORS_ARRAY = np.array(
    [SEG_COLORS[i] for i in range(len(SEG_COLORS))],
    dtype=np.uint8
)

def colorize_mask(mask):
    """将 label mask 转为 RGB 彩色图"""
    return COLORS_ARRAY[mask]

def get_legend_patches():
    return [
        mpatches.Patch(color=np.array(SEG_COLORS[i]) / 255,
                       label=ID2LABEL[i])
        for i in range(len(ID2LABEL))
    ]

def plot_seg_results(images, gt_masks, pred_masks,
                     save_path=None, title='Segmentation Results'):
    """
    images:     list of np.array (H, W, 3)
    gt_masks:   list of np.array (H, W)
    pred_masks: list of np.array (H, W)
    """
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = [axes]

    for row, (img, gt, pred) in enumerate(zip(images, gt_masks, pred_masks)):
        overlay = (img * 0.5 + colorize_mask(pred) * 0.5).astype(np.uint8)
        axes[row][0].imshow(img)
        axes[row][0].set_title('Original', fontsize=9)
        axes[row][0].axis('off')
        axes[row][1].imshow(colorize_mask(gt))
        axes[row][1].set_title('Ground Truth', fontsize=9)
        axes[row][1].axis('off')
        axes[row][2].imshow(overlay)
        axes[row][2].set_title('Prediction Overlay', fontsize=9)
        axes[row][2].axis('off')

    fig.legend(handles=get_legend_patches(),
               loc='lower center', ncol=5, fontsize=10)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"保存至：{save_path}")
    plt.show()

def plot_training_curves(train_losses, val_mious, val_epochs,
                         save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(val_epochs, train_losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_epochs, val_mious, 'r-o', linewidth=2, markersize=6)
    ax2.axhline(y=max(val_mious), color='gray', linestyle='--',
                alpha=0.7, label=f'Best: {max(val_mious):.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Validation mIoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('SegFormer Training on nuScenes Mini', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()