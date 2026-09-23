# infer_seg.py
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import CKPT_DIR, OUTPUT_DIR, IMG_DIR, MASK_DIR
from models.segformer import get_processor, load_trained_model
from utils.visualize import plot_seg_results


def run_inference(num_samples=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    ckpt_path = f'{CKPT_DIR}/segformer_best.pth'
    processor = get_processor()
    model     = load_trained_model(ckpt_path, device=device)

    # 取样本
    img_files = sorted([f for f in os.listdir(IMG_DIR)
                        if f.endswith('.jpg')])[:num_samples]

    images, gt_masks, pred_masks = [], [], []

    model.eval()
    for fname in img_files:
        token = fname.replace('.jpg', '')
        img   = Image.open(f'{IMG_DIR}/{fname}').convert('RGB')
        gt    = np.array(Image.open(f'{MASK_DIR}/{token}.png'))

        # 推理
        encoding     = processor(images=img, return_tensors='pt')
        pixel_values = encoding['pixel_values'].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        upsampled = F.interpolate(
            outputs.logits,
            size=(900, 1600),
            mode='bilinear', align_corners=False
        )
        pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        images.append(np.array(img))
        gt_masks.append(gt)
        pred_masks.append(pred)

    # 可视化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_seg_results(
        images, gt_masks, pred_masks,
        save_path=f'{OUTPUT_DIR}/inference_results.png',
        title=f'SegFormer Inference Results'
    )
    print("推理完成")


if __name__ == '__main__':
    run_inference()