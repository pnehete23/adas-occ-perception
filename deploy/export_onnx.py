# deploy/export_onnx.py
import os
import sys
import torch
import numpy as np

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import CKPT_DIR, OUTPUT_DIR, IMAGE_SIZE
from models.segformer import load_trained_model, get_processor


def export_to_onnx(ckpt_path, onnx_path, image_size=512):
    """
    把训练好的 SegFormer 导出为 ONNX 格式
    """
    device = torch.device('cpu')  # ONNX 导出用 CPU
    print(f"加载模型：{ckpt_path}")
    model = load_trained_model(ckpt_path, device='cpu')
    model.eval()

    # 构造 dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    print(f"导出 ONNX：{onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits':       {0: 'batch_size'},
        }
    )
    print("ONNX 导出完成")

    # 验证 ONNX 模型
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print(f"ONNX 模型验证通过")

    # 打印模型大小
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"模型大小：{size_mb:.1f} MB")
    return onnx_path


def verify_onnx_output(ckpt_path, onnx_path, image_size=512):
    """
    对比 PyTorch 和 ONNX 的输出是否一致
    """
    import onnxruntime as ort

    device = torch.device('cpu')
    model  = load_trained_model(ckpt_path, device='cpu')
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    # PyTorch 推理
    with torch.no_grad():
        pt_output = model(pixel_values=dummy_input).logits.numpy()

    # ONNX 推理
    sess = ort.InferenceSession(onnx_path,
                                providers=['CPUExecutionProvider'])
    onnx_output = sess.run(
        None, {'pixel_values': dummy_input.numpy()})[0]

    # 对比输出
    max_diff = np.abs(pt_output - onnx_output).max()
    print(f"PyTorch vs ONNX 最大误差：{max_diff:.6f}")
    if max_diff < 1e-3:
        print("✅ 输出一致，ONNX 导出正确")
    else:
        print("⚠️  输出差异较大，请检查")

    return max_diff


if __name__ == '__main__':
    ckpt_path = f'{CKPT_DIR}/segformer_best.pth'
    onnx_path = f'{CKPT_DIR}/segformer.onnx'

    # 安装依赖
    os.system('pip install -q onnx onnxruntime')

    # 导出
    export_to_onnx(ckpt_path, onnx_path)

    # 验证
    verify_onnx_output(ckpt_path, onnx_path)