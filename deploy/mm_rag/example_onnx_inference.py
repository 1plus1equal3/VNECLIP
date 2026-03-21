"""
Complete example: Convert models to ONNX and perform inference.
"""

import os
import sys
from PIL import Image

# First, convert models to ONNX
print("=" * 60)
print("STEP 1: Converting models to ONNX format")
print("=" * 60)

try:
    from convert_to_onnx import convert_vision_tower_to_onnx, convert_text_tower_to_onnx
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Convert vision tower
    convert_vision_tower_to_onnx(
        weight_path="./weight/vision_tower.pth",
        output_path="./weight/vision_tower_onnx.onnx",
        device=device
    )
    
    # Convert text tower
    convert_text_tower_to_onnx(
        weight_path="./weight/text_tower.pth",
        output_path="./weight/text_tower_onnx.onnx",
        device=device
    )
    
    print("\n✓ Models successfully converted to ONNX!")
    
except Exception as e:
    print(f"✗ Error during conversion: {e}")
    print("Continuing with existing ONNX models if available...\n")


# Now perform inference with ONNX models
print("\n" + "=" * 60)
print("STEP 2: Inference with ONNX models")
print("=" * 60 + "\n")

try:
    from retrieval_onnx import MultimodalRetrievalONNX
    
    # Initialize retrieval system with ONNX models
    print("Initializing ONNX-based retrieval system...")
    retrieval_client = MultimodalRetrievalONNX(
        db_path="./clip_db",
        collection_name="clip_retrieval",
        vision_tower_onnx="./weight/vision_tower_onnx.onnx",
        text_tower_onnx="./weight/text_tower_onnx.onnx",
        device='cuda',
        use_onnx=True
    )
    print("✓ Retrieval system initialized\n")
    
    # Test 1: Text query
    print("-" * 60)
    print("Test 1: Text Query")
    print("-" * 60)
    text_query = "Một con mèo đang nằm trên bàn."
    print(f"Query: {text_query}")
    
    results = retrieval_client.search(text_query, top_k=5)
    print("Results:")
    retrieval_client.visualize(results)
    
    # Test 2: Image query
    print("\n" + "-" * 60)
    print("Test 2: Image Query")
    print("-" * 60)
    
    image_path = "/root/Project/brick_vidgen/vnclip/deploy/sample/cat.jpg"
    if os.path.exists(image_path):
        image_query = Image.open(image_path).convert('RGB')
        print(f"Query image: {image_path}")
        
        results = retrieval_client.search(image_query, top_k=5)
        print("Results:")
        retrieval_client.visualize(results)
    else:
        print(f"Sample image not found at {image_path}")
    
    print("\n✓ Inference completed successfully!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure onnxruntime is installed: pip install onnxruntime")
except Exception as e:
    print(f"✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()


# Performance comparison
print("\n" + "=" * 60)
print("STEP 3: Performance Comparison")
print("=" * 60 + "\n")

try:
    import time
    import numpy as np
    from onnx_inference import ONNXVisionEncoder
    from tower import build_vision_tower
    from PIL import Image
    import torch
    
    # Create test images
    test_images = [Image.new('RGB', (224, 224)) for _ in range(10)]
    
    # ONNX inference
    print("Testing ONNX Vision Encoder performance...")
    onnx_encoder = ONNXVisionEncoder("./weight/vision_tower_onnx.onnx", 'cuda')
    
    start = time.time()
    for _ in range(5):
        _ = onnx_encoder.encode(test_images)
    onnx_time = (time.time() - start) / 5
    print(f"ONNX inference time (10 images): {onnx_time*1000:.2f}ms")
    
    # PyTorch inference for comparison
    print("\nTesting PyTorch Vision Encoder performance...")
    torch_encoder = build_vision_tower("./weight/vision_tower.pth", 'cuda')
    
    with torch.no_grad():
        start = time.time()
        for _ in range(5):
            images_tensor = torch.stack([
                torch.randn(3, 224, 224) for _ in test_images
            ]).cuda()
            _ = torch_encoder(images_tensor)
        pytorch_time = (time.time() - start) / 5
    print(f"PyTorch inference time (10 images): {pytorch_time*1000:.2f}ms")
    
    speedup = pytorch_time / onnx_time
    print(f"\n✓ ONNX Speedup: {speedup:.2f}x faster than PyTorch")
    
except Exception as e:
    print(f"Performance comparison skipped: {e}")
