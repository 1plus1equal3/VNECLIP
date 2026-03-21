"""
Convert both Vision Tower and Text Tower to ONNX format for edge deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from model.model import ProjectionHead
from model.convnext import ConvNeXt, convnext_small
from model.phobert import PhoBERT
from tower import INFER_TRANSFORM, tokenizer
import os

# ==================== Vision Tower Wrapper ====================
class VisionEncoderONNX(nn.Module):
    """Wrapper for vision tower that can be exported to ONNX."""
    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, images):
        features = self.encoder(images)
        projected = self.projection(features)
        return projected


# ==================== Text Tower Wrapper ====================
class TextEncoderONNX(nn.Module):
    """Wrapper for text tower that can be exported to ONNX."""
    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, input_ids, attention_mask):
        # Get features from PhoBERT
        outputs = self.encoder.model(input_ids, attention_mask=attention_mask)
        features = outputs[0][:, 0, :]  # CLS token embedding
        
        # Project the features
        projected = self.projection(features)
        return projected


# ==================== Conversion Functions ====================
def convert_vision_tower_to_onnx(
    weight_path="../weight/vision_tower.pth",
    output_path="../weight/vision_tower.onnx",
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
    vision_tower = VisionEncoderONNX(convnext, projection)
    
    # Load weights
    print("Loading weights...")
    weight = torch.load(weight_path, map_location=device)
    
    # Filter out keys that don't belong to the vision tower
    filtered_weight = {}
    for k, v in weight.items():
        # Remove 'encoder.' and 'projection.' prefixes if present
        if k.startswith('encoder.'):
            filtered_weight[k] = v
        elif k.startswith('projection.'):
            filtered_weight[k] = v
        else:
            filtered_weight[k] = v
    
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
    
    print(f"✓ Vision ONNX model successfully created: {output_path}")
    print(f"✓ Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    

def convert_text_tower_to_onnx(
    weight_path="../weight/text_tower.pth",
    output_path="../weight/text_tower.onnx",
    device='cpu'
):
    """
    Convert text tower to ONNX format.
    
    Args:
        weight_path: Path to the PyTorch weights
        output_path: Path where ONNX model will be saved
        device: Device to use for conversion ('cpu' or 'cuda')
    """
    print(f"\nConverting text tower from {weight_path}...")
    print(f"Output ONNX model: {output_path}")
    
    # Build model architecture
    phobert = PhoBERT(model_name="vinai/phobert-base-v2")
    projection = ProjectionHead(768, 384, 0.1)
    text_tower = TextEncoderONNX(phobert, projection)
    
    # Load weights
    print("Loading weights...")
    weight = torch.load(weight_path, map_location=device)
    text_tower.load_state_dict(weight, strict=False)
    text_tower.eval()
    text_tower.to(device)
    
    # Prepare dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (1, 50), device=device)
    dummy_attention_mask = torch.ones((1, 50), dtype=torch.long, device=device)
    
    # Export to ONNX
    print("Exporting to ONNX format...")
    torch.onnx.export(
        text_tower,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['embeddings'],
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'embeddings': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Text ONNX model successfully created: {output_path}")
    print(f"✓ Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    # Create ONNX directory if not exists
    onnx_dir = "../weight"
    os.makedirs(onnx_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Convert both towers
    convert_vision_tower_to_onnx(device=device)
    convert_text_tower_to_onnx(device=device)
    
    print("\n✅ All models successfully converted to ONNX!")
