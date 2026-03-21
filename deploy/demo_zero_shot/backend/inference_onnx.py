"""
Inference using ONNX Vision Tower - optimized for edge deployment.
"""

import onnxruntime as ort
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from config import PROMPT_EMBEDDING_PATH, PROMPT_PATH, ONNX_MODEL_PATH
import os

INFER_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# Preload prompt embeddings and text prompts
with open(PROMPT_EMBEDDING_PATH, 'rb') as f:
    PROMPT_EMBEDDINGS = np.load(f)
with open(PROMPT_PATH, 'r') as f:
    PROMPTS = [line.strip() for line in f.readlines()]

class ONNXVisionTower:
    """Vision tower inference using ONNX model."""
    
    def __init__(self, onnx_model_path=ONNX_MODEL_PATH, providers=None):
        """
        Initialize ONNX vision tower.
        
        Args:
            onnx_model_path: Path to ONNX model
            providers: ONNX Runtime execution providers. 
                      Options: ['CUDAExecutionProvider', 'CPUExecutionProvider']
                      If None, auto-detects best available
        """
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")
        
        if providers is None:
            # Auto-detect best provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print(f"Loading ONNX model from {onnx_model_path}...")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Print model info
        providers_used = self.session.get_providers()
        print(f"✓ Using providers: {providers_used}")
        print(f"✓ Model loaded successfully")
    
    def encode_images(self, images_tensor):
        """
        Encode images to embeddings.
        
        Args:
            images_tensor: torch.Tensor of shape (batch_size, 3, 224, 224) or numpy array
        
        Returns:
            numpy array of shape (batch_size, 384) containing embeddings
        """
        # Convert to numpy if torch tensor
        if isinstance(images_tensor, torch.Tensor):
            images_np = images_tensor.cpu().numpy()
        else:
            images_np = images_tensor
        
        # Ensure float32
        images_np = images_np.astype(np.float32)
        
        # Run inference
        embeddings = self.session.run(
            [self.output_name],
            {self.input_name: images_np}
        )[0]
        
        return embeddings
    
    def encode_single_image(self, image_path):
        """
        Encode a single image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            numpy array of shape (384,) containing embedding
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = INFER_TRANSFORM(image).unsqueeze(0)
        embeddings = self.encode_images(image_tensor)
        return embeddings[0]
    
    def encode_batch_images(self, image_paths):
        """
        Encode multiple images from file paths.
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            numpy array of shape (len(image_paths), 384) containing embeddings
        """
        images = [INFER_TRANSFORM(Image.open(p).convert('RGB')) for p in image_paths]
        images_tensor = torch.stack(images)
        embeddings = self.encode_images(images_tensor)
        return embeddings


def load_prompt_embeddings():
    """Load precomputed prompt embeddings."""
    with open(PROMPT_EMBEDDING_PATH, 'rb') as f:
        prompt_embeddings = np.load(f)
    with open(PROMPT_PATH, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompt_embeddings, prompts


def similarity_search(image_embedding, prompt_embeddings, top_k=5):
    """
    Find most similar prompts using cosine similarity.
    
    Args:
        image_embedding: numpy array of shape (384,)
        prompt_embeddings: numpy array of shape (n_prompts, 384)
        top_k: Number of top results to return
    
    Returns:
        List of tuples (score, index)
    """
    # Normalize embeddings
    image_emb_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
    prompt_emb_norm = prompt_embeddings / (np.linalg.norm(prompt_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity
    similarities = np.dot(prompt_emb_norm, image_emb_norm)
    
    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    
    return list(zip(top_scores, top_indices))


def zero_shot_classify(image, vision_tower, prompt_embeddings=PROMPT_EMBEDDINGS, prompts=PROMPTS, top_k=3):
    """
    Zero-shot image classification using ONNX vision tower.
    Same signature as PyTorch version for compatibility.
    
    Args:
        image: PIL.Image or path to image file
        vision_tower: ONNXVisionTower instance
        prompt_embeddings: numpy array of shape (n_prompts, 384), defaults to preloaded PROMPT_EMBEDDINGS
        prompts: List of prompt strings, defaults to preloaded PROMPTS
        top_k: Number of top results to return (default: 3)
    
    Returns:
        List of top-k prompt strings
    """
    # Encode image
    if isinstance(image, str):
        image_embedding = vision_tower.encode_single_image(image)
    else:
        # Assume PIL.Image
        image_tensor = INFER_TRANSFORM(image).unsqueeze(0)
        image_embedding = vision_tower.encode_images(image_tensor)[0]
    
    # Find similar prompts
    results = similarity_search(image_embedding, prompt_embeddings, top_k=top_k)
    
    # Extract prompt strings
    retrieved_prompts = [prompts[idx] for _, idx in results]
    
    return retrieved_prompts


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("ONNX Vision Tower Inference Example")
    print("=" * 60)
    
    # Initialize ONNX vision tower
    vision_tower = ONNXVisionTower()
    
    # Load prompt embeddings
    prompt_embeddings, prompts = load_prompt_embeddings()
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Example: encode a test image
    print("\nExample: Encoding test image...")
    test_image = "/root/Project/brick_vidgen/vnclip/deploy/data/motorbike.jpg"  # Change path as needed
    
    if os.path.exists(test_image):
        embedding = vision_tower.encode_single_image(test_image)
        results = similarity_search(embedding, prompt_embeddings, top_k=5)
        
        print(f"\nTop 5 matching prompts:")
        for i, (score, idx) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {prompts[idx]}")
    else:
        print(f"Test image not found at {test_image}")
