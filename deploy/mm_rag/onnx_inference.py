"""
ONNX Runtime inference for vision and text encoders.
Provides fast inference on edge devices.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from underthesea import word_tokenize
from transformers import AutoTokenizer
import onnxruntime as ort

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Image preprocessing
INFER_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


class ONNXVisionEncoder:
    """ONNX-based Vision Encoder for image embedding."""
    
    def __init__(self, model_path="./weight/vision_tower_onnx.onnx", device='cpu'):
        """
        Initialize ONNX vision encoder.
        
        Args:
            model_path: Path to the ONNX model
            device: 'cpu' or 'cuda'
        """
        print(f"Loading ONNX vision encoder from {model_path}...")
        
        # Set up execution provider
        providers = []
        if device == 'cuda':
            providers.append(('CUDAExecutionProvider', {'device_id': 0}))
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        print("✓ Vision encoder loaded successfully")
    
    def encode(self, images):
        """
        Encode images to embeddings.
        
        Args:
            images: PIL Image or list of PIL Images
        
        Returns:
            numpy array of shape (batch_size, 384)
        """
        if isinstance(images, Image.Image):
            images = [images]
        elif not isinstance(images, list):
            raise ValueError("Input must be a PIL Image or a list of PIL Images.")
        
        # Preprocess images
        images_tensor = torch.stack([
            INFER_TRANSFORM(img.convert('RGB')) for img in images
        ])
        images_np = images_tensor.numpy()
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        embeddings = self.session.run([output_name], {input_name: images_np})
        
        # Normalize embeddings
        embeddings = embeddings[0]
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        
        return embeddings


class ONNXTextEncoder:
    """ONNX-based Text Encoder for text embedding."""
    
    def __init__(self, model_path="./weight/text_tower_onnx.onnx", device='cpu'):
        """
        Initialize ONNX text encoder.
        
        Args:
            model_path: Path to the ONNX model
            device: 'cpu' or 'cuda'
        """
        print(f"Loading ONNX text encoder from {model_path}...")
        
        # Set up execution provider
        providers = []
        if device == 'cuda':
            providers.append(('CUDAExecutionProvider', {'device_id': 0}))
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        print("✓ Text encoder loaded successfully")
    
    def encode(self, texts):
        """
        Encode texts to embeddings.
        
        Args:
            texts: str or list of strings
        
        Returns:
            numpy array of shape (batch_size, 384)
        """
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError("Input must be a string or a list of strings.")
        
        # Tokenize texts
        seg_texts = [word_tokenize(text, format="text") for text in texts]
        input_ids_list = []
        attention_masks_list = []
        
        for text in seg_texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=50
            )
            input_ids_list.append(inputs["input_ids"].numpy())
            attention_masks_list.append(inputs["attention_mask"].numpy())
        
        input_ids = np.vstack(input_ids_list).astype(np.int64)
        attention_mask = np.vstack(attention_masks_list).astype(np.int64)
        
        # Run inference
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_name = self.session.get_outputs()[0].name
        
        embeddings = self.session.run(
            [output_name],
            {
                input_names[0]: input_ids,
                input_names[1]: attention_mask
            }
        )
        
        # Normalize embeddings
        embeddings = embeddings[0]
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        
        return embeddings


# ==================== Convenience Functions ====================

def load_onnx_vision_encoder(model_path="./weight/vision_tower_onnx.onnx", device='cpu'):
    """Load and return ONNX vision encoder."""
    return ONNXVisionEncoder(model_path, device)


def load_onnx_text_encoder(model_path="./weight/text_tower_onnx.onnx", device='cpu'):
    """Load and return ONNX text encoder."""
    return ONNXTextEncoder(model_path, device)


def encode_images(vision_encoder, images):
    """Encode images using ONNX vision encoder."""
    return vision_encoder.encode(images)


def encode_texts(text_encoder, texts):
    """Encode texts using ONNX text encoder."""
    return text_encoder.encode(texts)
