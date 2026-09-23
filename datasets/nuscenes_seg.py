# datasets/nuscenes_seg.py
import os
from torch.utils.data import Dataset
from PIL import Image
import sys

sys.path.insert(0, '/content/drive/MyDrive/adas_occ_project')
from config import IMG_DIR, MASK_DIR, TRAIN_SPLIT

class NuScenesSegDataset(Dataset):
    def __init__(self, processor, split='train',
                 img_dir=IMG_DIR, mask_dir=MASK_DIR):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.processor = processor

        from nuscenes.nuscenes import NuScenes
        from config import DATA_ROOT, NUSCENES_VERSION

        nusc = NuScenes(version=NUSCENES_VERSION,
                        dataroot=DATA_ROOT, verbose=False)

        # Scene-level split：前8个scene训练，后2个scene验证
        n_scenes  = len(nusc.scene)
        split_idx = int(n_scenes * TRAIN_SPLIT)
        train_scenes = set(s['token'] for s in nusc.scene[:split_idx])
        val_scenes   = set(s['token'] for s in nusc.scene[split_idx:])

        def get_scene_token(sample_token):
            sample = nusc.get('sample', sample_token)
            return nusc.get('scene', sample['scene_token'])['token']

        all_tokens = sorted([f.replace('.jpg', '')
                            for f in os.listdir(img_dir)
                            if f.endswith('.jpg')])

        if split == 'train':
            self.tokens = [t for t in all_tokens
                        if get_scene_token(t) in train_scenes]
        else:
            self.tokens = [t for t in all_tokens
                        if get_scene_token(t) in val_scenes]
        print(f"[NuScenesSegDataset] {split}: {len(self.tokens)} samples")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        img  = Image.open(f'{self.img_dir}/{token}.jpg').convert('RGB')
        mask = Image.open(f'{self.mask_dir}/{token}.png')
        encoding = self.processor(
            images=img,
            segmentation_maps=mask,
            return_tensors='pt'
        )
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'labels':       encoding['labels'].squeeze(),
        }