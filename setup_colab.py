# setup_colab.py
# 每次重启 Colab 只需运行这一个文件
import os
import subprocess
import sys

def run(cmd):
    print(f">>> {cmd}")
    os.system(cmd)

print("=" * 50)
print("Step 1: 安装依赖")
print("=" * 50)
run("pip install -q numpy==1.26.4 --force-reinstall")
run("pip install -q transformers accelerate")
run("pip install -q nuscenes-devkit pyquaternion")

print("\n" + "=" * 50)
print("Step 2: 解压数据集")
print("=" * 50)
DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
DATA_ROOT  = '/content/nuscenes'

if not os.path.exists(f'{DATA_ROOT}/v1.0-mini'):
    os.makedirs(DATA_ROOT, exist_ok=True)
    run(f'tar -xzf {DRIVE_ROOT}/v1.0-mini.tgz -C {DATA_ROOT}')
    run(f'unzip -q {DRIVE_ROOT}/nuScenes-map-expansion-v1.3.zip -d {DATA_ROOT}')
    print("解压完成")
else:
    print("数据集已存在，跳过解压")

print("\n" + "=" * 50)
print("Step 3: 生成分割 Mask")
print("=" * 50)
SEG_DATA_DIR = '/content/seg_data'

if not os.path.exists(f'{SEG_DATA_DIR}/images'):
    from nuscenes.nuscenes import NuScenes
    sys.path.insert(0, DRIVE_ROOT)
    from utils.mask_generator import generate_all_masks
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=False)
    generate_all_masks(nusc)
else:
    print("Mask 已存在，跳过生成")

print("\n" + "=" * 50)
print("✅ 环境准备完成！可以开始训练。")
print("=" * 50)