# inference benchmark
import os
import torch
import time
import random
import gc
import clip
from PIL import Image
from tqdm import tqdm
from inference import build_vision_tower, INFER_TRANSFORM
from inference_onnx import ONNXVisionTower, INFER_TRANSFORM as ONNX_INFER_TRANSFORM
from config import ORIGINAL_CLIP_PATH, ONNX_MODEL_PATH


def cleanup_memory():
    """Clear GPU and CPU memory after benchmark."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clip_benchmark(images, device="cpu", iters=100):
    model, preprocess = clip.load(ORIGINAL_CLIP_PATH, device=device)
    latencies = []
    for _ in tqdm(range(iters)):
        image = random.choice(images)
        image = preprocess(image).unsqueeze(0).to(device)
        start_time = time.time()
        with torch.no_grad():
            _ = model.encode_image(image.to(device))
        end_time = time.time()
        latencies.append(end_time - start_time)
    print(f"Average latency over {iters} iterations: {sum(latencies) / len(latencies):.4f} seconds")
    print(f"Median latency: {sorted(latencies)[len(latencies) // 2]:.4f} seconds")
    print(f"Approximate FPS: {1 / (sum(latencies) / len(latencies)):.2f}")


def vnclip_benchmark(images, device="cpu", iters=100):
    vision_tower = build_vision_tower(device=device)
    latencies = []
    for _ in tqdm(range(iters)):
        image = random.choice(images)
        image = INFER_TRANSFORM(image.convert('RGB')).unsqueeze(0)
        start_time = time.time()
        with torch.no_grad():
            _ = vision_tower(image.to(device))
        end_time = time.time()
        latencies.append(end_time - start_time)
    print(f"Average latency over {iters} iterations: {sum(latencies) / len(latencies):.4f} seconds")
    print(f"Median latency: {sorted(latencies)[len(latencies) // 2]:.4f} seconds")
    print(f"Approximate FPS: {1 / (sum(latencies) / len(latencies)):.2f} frames per second")


def vnclip_onnx_benchmark(images, device="cpu", iters=100):
    """Benchmark VNCLIP ONNX model for optimized inference."""
    # Determine ONNX providers based on device
    if device == "cuda":
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    vision_tower = ONNXVisionTower(onnx_model_path=ONNX_MODEL_PATH, providers=providers)
    latencies = []
    for _ in tqdm(range(iters)):
        image = random.choice(images)
        image = ONNX_INFER_TRANSFORM(image.convert('RGB')).unsqueeze(0)
        start_time = time.time()
        _ = vision_tower.encode_images(image)
        end_time = time.time()
        latencies.append(end_time - start_time)
    print(f"Average latency over {iters} iterations: {sum(latencies) / len(latencies):.4f} seconds")
    print(f"Median latency: {sorted(latencies)[len(latencies) // 2]:.4f} seconds")
    print(f"Approximate FPS: {1 / (sum(latencies) / len(latencies)):.2f} frames per second")

if __name__ == "__main__":
    # Load sample images for benchmarking
    image_dir = "/root/deploy/sample/"
    images = [Image.open(os.path.join(image_dir, fname)) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.png'))]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run benchmarks
    print("Running CLIP benchmark...")
    clip_benchmark(images, device=device, iters=100)
    cleanup_memory()
    
    print("\n" + "="*50 + "\n")
    print("Running VN-CLIP vision tower benchmark...")
    vnclip_benchmark(images, device=device, iters=100)
    cleanup_memory()
    
    print("\n" + "="*50 + "\n")
    print("Running VN-CLIP ONNX benchmark...")
    vnclip_onnx_benchmark(images, device=device, iters=100)
    cleanup_memory()
    
    print("\n" + "="*50)
    print("All benchmarks completed!")