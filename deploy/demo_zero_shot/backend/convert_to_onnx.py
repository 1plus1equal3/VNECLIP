"""
Convert Vision Tower (ConvNeXt + Projection Head) to ONNX format for edge deployment.
"""

import torch
import torch.nn as nn
from model import convnext_small, ProjectionHead
from inference import EncoderTower
from config import VISION_TOWER_WEIGHT_PATH
import os

def convert_vision_tower_to_onnx(
    weight_path=VISION_TOWER_WEIGHT_PATH,
    output_path="vision_tower.onnx",
    device='cpu'
):
    """
    Convert vision tower to ONNX format.
    
    Args:
        weight_path: Path to the PyTorch weights
        output_path: Path where ONNX model will be saved
        device: Device to use for conversion ('cpu' or 'cuda')
    """
    print(f"Converting vision tower from {weight_path}...")
    print(f"Output ONNX model: {output_path}")
    
    # Build model architecture
    convnext = convnext_small(pretrained=False, weight_path=None)
    projection = ProjectionHead(768, 384, 0.1)
    vision_tower = EncoderTower(convnext, projection)
    
    # Load weights
    print("Loading weights...")
    weight = torch.load(weight_path, map_location=device)
    
    # Filter out logit_scale if it exists (not needed for inference)
    filtered_weight = {k: v for k, v in weight.items() if k != 'logit_scale'}
    vision_tower.load_state_dict(filtered_weight, strict=False)
    vision_tower.eval()
    vision_tower.to(device)
    
    # Prepare dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Export to ONNX
    print("Exporting to ONNX format...")
    torch.onnx.export(
        vision_tower,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=['embeddings'],
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamic_axes={
            'images': {0: 'batch_size'},
            'embeddings': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX model successfully created: {output_path}")
    print(f"✓ Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    

if __name__ == "__main__":
    # Convert to ONNX
    convert_vision_tower_to_onnx()
