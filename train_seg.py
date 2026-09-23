# train_seg.py
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import (BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS,
                    LR, WEIGHT_DECAY, CKPT_DIR, OUTPUT_DIR)
from models.segformer import get_model, get_processor
from datasets.nuscenes_seg import NuScenesSegDataset
from utils.visualize import plot_training_curves


def compute_miou(preds, labels, num_classes=5):
    ious = []
    preds  = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    for cls in range(num_classes):
        pred_cls  = preds  == cls
        label_cls = labels == cls
        intersection = (pred_cls & label_cls).sum()
        union        = (pred_cls | label_cls).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型和数据
    processor = get_processor()
    model     = get_model().to(device)

    train_dataset = NuScenesSegDataset(processor, split='train')
    val_dataset   = NuScenesSegDataset(processor, split='val')
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=NUM_WORKERS)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 断点续训：如果已有权重则加载
    ckpt_path = f'{CKPT_DIR}/segformer_best.pth'
    best_miou = 0.0
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"加载已有权重：{ckpt_path}")

    train_losses, val_mious, val_epochs = [], [], []

    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels       = batch['labels'].to(device)
            outputs      = model(pixel_values=pixel_values, labels=labels)
            loss         = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证（每5个epoch）
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_mious = []
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch['pixel_values'].to(device)
                    labels       = batch['labels'].to(device)
                    outputs      = model(pixel_values=pixel_values)
                    upsampled    = torch.nn.functional.interpolate(
                        outputs.logits,
                        size=labels.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                    preds = upsampled.argmax(dim=1)
                    for p, l in zip(preds, labels):
                        all_mious.append(compute_miou(p, l))

            val_miou = np.mean(all_mious)
            val_mious.append(val_miou)
            val_epochs.append(epoch + 1)

            if val_miou > best_miou:
                best_miou = val_miou
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

            print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val mIoU: {val_miou:.4f} | "
                  f"Best: {best_miou:.4f}")

    # 训练曲线
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training_curves(
        train_losses, val_mious, val_epochs,
        save_path=f'{OUTPUT_DIR}/training_curves.png'
    )
    print(f"\n训练完成！最佳 mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    train()