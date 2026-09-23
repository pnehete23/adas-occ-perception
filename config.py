# config.py
import os

# 路径配置 
import os

# 自动检测运行环境
def _get_project_root():
    if 'ADAS_PROJECT_ROOT' in os.environ:
        return os.environ['ADAS_PROJECT_ROOT']
    colab_path = '/content/drive/MyDrive/adas_occ_project'
    if os.path.exists(colab_path):
        return colab_path
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = _get_project_root()

# 路径配置
DRIVE_ROOT    = PROJECT_ROOT
DATA_ROOT     = '/content/nuscenes' if os.path.exists('/content/nuscenes') \
                else os.path.join(PROJECT_ROOT, 'data/nuscenes')
SEG_DATA_DIR  = '/content/seg_data' if os.path.exists('/content/seg_data') \
                else os.path.join(PROJECT_ROOT, 'data/seg_data')
IMG_DIR       = f'{SEG_DATA_DIR}/images'
MASK_DIR      = f'{SEG_DATA_DIR}/masks'
CKPT_DIR      = f'{PROJECT_ROOT}/checkpoints'
OUTPUT_DIR    = f'{PROJECT_ROOT}/outputs'

# 数据配置 
NUSCENES_VERSION = 'v1.0-mini'
CAM_CHANNEL      = 'CAM_FRONT'
TRAIN_SPLIT      = 0.8

# 类别配置 
ID2LABEL = {
    0: 'background',
    1: 'vehicle',
    2: 'cyclist',
    3: 'pedestrian',
    4: 'obstacle',
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_CLASSES = len(ID2LABEL)

CAT_MAP = {
    'vehicle.car': 1, 'vehicle.truck': 1, 'vehicle.bus': 1,
    'vehicle.motorcycle': 2, 'vehicle.bicycle': 2,
    'human.pedestrian.adult': 3, 'human.pedestrian.child': 3,
    'human.pedestrian.construction_worker': 3,
    'human.pedestrian.police_officer': 3,
    'movable_object.barrier': 4, 'movable_object.trafficcone': 4,
}

# 模型配置 
PRETRAINED_MODEL = 'nvidia/segformer-b2-finetuned-ade-512-512'
IMAGE_SIZE       = 512

# 训练配置 
BATCH_SIZE   = 4
NUM_WORKERS  = 2
NUM_EPOCHS   = 30
LR           = 6e-5
WEIGHT_DECAY = 0.01

# BEV 配置 
BEV_RANGE  = 40    # 前方40米
BEV_SIZE   = 256   # 输出分辨率
CAM_HEIGHT = 1.8   # 相机离地高度（米）

# 可视化颜色
SEG_COLORS = {
    0: [0,   0,   0  ],  # background
    1: [255, 0,   0  ],  # vehicle
    2: [0,   0,   255],  # cyclist
    3: [0,   255, 0  ],  # pedestrian
    4: [255, 255, 0  ],  # obstacle
}