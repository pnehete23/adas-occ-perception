# utils/mask_generator.py
import os
import sys
import numpy as np
from PIL import Image

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import (DATA_ROOT, CAT_MAP, CAM_CHANNEL,
                    IMG_DIR, MASK_DIR)


def project_box_to_mask(nusc, sample_token, cam_channel):
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion

    sample   = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cs  = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    ego = nusc.get('ego_pose', cam_data['ego_pose_token'])

    img  = Image.open(DATA_ROOT + '/' + cam_data['filename'])
    W, H = img.size
    mask = np.zeros((H, W), dtype=np.uint8)

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
        box.translate(-np.array(cs['translation']))
        box.rotate(Quaternion(cs['rotation']).inverse)

        if box.center[2] < 0.1:
            continue

        corners = view_points(
            box.corners(),
            np.array(cs['camera_intrinsic']),
            normalize=True
        )
        x1 = int(np.clip(corners[0].min(), 0, W))
        x2 = int(np.clip(corners[0].max(), 0, W))
        y1 = int(np.clip(corners[1].min(), 0, H))
        y2 = int(np.clip(corners[1].max(), 0, H))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = label

    return img, mask


def generate_all_masks(nusc, img_dir=IMG_DIR, mask_dir=MASK_DIR):
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i, sample in enumerate(nusc.sample):
        img, mask = project_box_to_mask(nusc, sample['token'], CAM_CHANNEL)
        token = sample['token']
        img.save(f'{img_dir}/{token}.jpg')
        Image.fromarray(mask).save(f'{mask_dir}/{token}.png')
        if i % 50 == 0:
            print(f'进度: {i+1}/{len(nusc.sample)}')

    print(f"完成：{len(nusc.sample)} 对图像/mask")