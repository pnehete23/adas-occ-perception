# deploy/benchmark.py
import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

DRIVE_ROOT = '/content/drive/MyDrive/adas_occ_project'
sys.path.insert(0, DRIVE_ROOT)

from config import CKPT_DIR, OUTPUT_DIR, IMAGE_SIZE
from models.segformer import load_trained_model


def benchmark_pytorch(ckpt_path, image_size=512,
                       num_runs=100, warmup=10):
    """
    测量 PyTorch 模型推理延迟
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch 推理设备: {device}")

    model = load_trained_model(ckpt_path, device=str(device))
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(pixel_values=dummy)

    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(pixel_values=dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    avg  = np.mean(latencies)
    std  = np.std(latencies)
    p50  = np.percentile(latencies, 50)
    p95  = np.percentile(latencies, 95)
    fps  = 1000 / avg

    print(f"\n── PyTorch ({device}) ──────────────────")
    print(f"  平均延迟:  {avg:.2f} ± {std:.2f} ms")
    print(f"  P50:       {p50:.2f} ms")
    print(f"  P95:       {p95:.2f} ms")
    print(f"  FPS:       {fps:.1f}")
    return {'avg': avg, 'std': std, 'p50': p50,
            'p95': p95, 'fps': fps, 'latencies': latencies}


def benchmark_onnx(onnx_path, image_size=512,
                   num_runs=100, warmup=10):
    """
    测量 ONNX Runtime 推理延迟
    """
    import onnxruntime as ort

    # 优先用 CUDA EP，否则用 CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    provider = sess.get_providers()[0]
    print(f"ONNX 推理 Provider: {provider}")

    dummy = np.random.randn(1, 3, image_size,
                            image_size).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {'pixel_values': dummy})

    # 计时
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        sess.run(None, {'pixel_values': dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    avg  = np.mean(latencies)
    std  = np.std(latencies)
    p50  = np.percentile(latencies, 50)
    p95  = np.percentile(latencies, 95)
    fps  = 1000 / avg

    print(f"\n── ONNX Runtime ({provider}) ──────────")
    print(f"  平均延迟:  {avg:.2f} ± {std:.2f} ms")
    print(f"  P50:       {p50:.2f} ms")
    print(f"  P95:       {p95:.2f} ms")
    print(f"  FPS:       {fps:.1f}")
    return {'avg': avg, 'std': std, 'p50': p50,
            'p95': p95, 'fps': fps, 'latencies': latencies}


def plot_benchmark_results(pt_results, onnx_results, save_path=None):
    """
    可视化延迟对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 延迟分布直方图
    axes[0].hist(pt_results['latencies'], bins=30,
                 alpha=0.7, label='PyTorch', color='steelblue')
    axes[0].hist(onnx_results['latencies'], bins=30,
                 alpha=0.7, label='ONNX', color='orange')
    axes[0].set_xlabel('Latency (ms)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Latency Distribution')
    axes[0].legend()

    # 2. 平均延迟对比柱状图
    methods = ['PyTorch\n(GPU)', 'ONNX\nRuntime']
    avgs    = [pt_results['avg'], onnx_results['avg']]
    stds    = [pt_results['std'], onnx_results['std']]
    colors  = ['steelblue', 'orange']
    bars = axes[1].bar(methods, avgs, yerr=stds,
                       color=colors, capsize=5, alpha=0.8)
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Average Latency Comparison')
    for bar, avg in zip(bars, avgs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{avg:.1f}ms', ha='center', fontsize=11)

    # 3. FPS 对比
    fpss   = [pt_results['fps'], onnx_results['fps']]
    bars2  = axes[2].bar(methods, fpss, color=colors, alpha=0.8)
    axes[2].set_ylabel('FPS')
    axes[2].set_title('Throughput (FPS)')
    for bar, fps in zip(bars2, fpss):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{fps:.1f}', ha='center', fontsize=11)

    # 加速比
    speedup = pt_results['avg'] / onnx_results['avg']
    fig.suptitle(
        f'SegFormer Inference Benchmark  |  '
        f'ONNX speedup: {speedup:.2f}x',
        fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存至：{save_path}")
    plt.show()
    return speedup


if __name__ == '__main__':
    ckpt_path = f'{CKPT_DIR}/segformer_best.pth'
    onnx_path = f'{CKPT_DIR}/segformer.onnx'

    os.system('pip install -q onnx onnxruntime-gpu')

    print("=" * 50)
    print("SegFormer 推理性能基准测试")
    print("=" * 50)

    pt_results   = benchmark_pytorch(ckpt_path)
    onnx_results = benchmark_onnx(onnx_path)

    speedup = plot_benchmark_results(
        pt_results, onnx_results,
        save_path=f'{OUTPUT_DIR}/benchmark_results.png'
    )

    print(f"\n{'='*50}")
    print(f"ONNX 加速比: {speedup:.2f}x")
    print(f"PyTorch FPS: {pt_results['fps']:.1f}")
    print(f"ONNX FPS:    {onnx_results['fps']:.1f}")
    print(f"{'='*50}")
    
    
    print("\n── CPU 对比测试 ────────────────────")
    pt_cpu_results = benchmark_pytorch(ckpt_path, num_runs=20, warmup=3)

    # ONNX CPU
    import onnxruntime as ort
    sess_cpu = ort.InferenceSession(onnx_path,
            providers=['CPUExecutionProvider'])
    dummy_np = np.random.randn(1, 3, 512, 512).astype(np.float32)

    # warmup
    for _ in range(3):
        sess_cpu.run(None, {'pixel_values': dummy_np})

    cpu_latencies = []
    for _ in range(20):
        start = time.perf_counter()
        sess_cpu.run(None, {'pixel_values': dummy_np})
        cpu_latencies.append((time.perf_counter() - start) * 1000)

    onnx_cpu_avg = np.mean(cpu_latencies)
    print(f"PyTorch CPU 平均延迟: {pt_cpu_results['avg']:.1f} ms")
    print(f"ONNX CPU 平均延迟:    {onnx_cpu_avg:.1f} ms")
    print(f"CPU 加速比: {pt_cpu_results['avg'] / onnx_cpu_avg:.2f}x")