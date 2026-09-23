# models/segformer.py
import torch
import sys
sys.path.insert(0, '/content/drive/MyDrive/adas_occ_project')
from config import (PRETRAINED_MODEL, NUM_CLASSES,
                    ID2LABEL, LABEL2ID, IMAGE_SIZE)
from transformers import (SegformerForSemanticSegmentation,
                           SegformerImageProcessor)


def get_processor():
    return SegformerImageProcessor(
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        size={'height': IMAGE_SIZE, 'width': IMAGE_SIZE},
        do_reduce_labels=False,
    )


def get_model(pretrained=True):
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    return model


def load_trained_model(ckpt_path, device='cuda'):
    from transformers import SegformerConfig
    config = SegformerConfig.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = SegformerForSemanticSegmentation(config)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"模型加载成功：{ckpt_path}")
    return model