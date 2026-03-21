"""
Updated MultimodalRetrieval system that supports both PyTorch and ONNX models.
"""

import os
from PIL import Image
from chroma_db import ImageVecDB
from embedder import text_embedding, image_embedding
from onnx_inference import (
    ONNXVisionEncoder, ONNXTextEncoder,
    encode_images, encode_texts
)


class MultimodalRetrievalONNX:
    """
    Multimodal Retrieval system using ONNX models for better edge device performance.
    """
    
    def __init__(
        self,
        db_path="./clip_db",
        collection_name="clip_retrieval",
        vision_tower_onnx="./weight/vision_tower_onnx.onnx",
        text_tower_onnx="./weight/text_tower_onnx.onnx",
        device='cpu',
        use_onnx=True
    ):
        """
        Initialize the retrieval system.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection in ChromaDB
            vision_tower_onnx: Path to ONNX vision model
            text_tower_onnx: Path to ONNX text model
            device: 'cpu' or 'cuda'
            use_onnx: Use ONNX models if True, else use PyTorch models
        """
        self.db = ImageVecDB(db_path, collection_name)
        self.device = device
        self.use_onnx = use_onnx
        
        if use_onnx:
            print("Initializing with ONNX models...")
            self.vision_tower = ONNXVisionEncoder(vision_tower_onnx, device)
            self.text_tower = ONNXTextEncoder(text_tower_onnx, device)
        else:
            print("Initializing with PyTorch models...")
            # Fall back to PyTorch if ONNX not available
            from tower import build_vision_tower, build_text_tower
            self.vision_tower = build_vision_tower(
                vision_tower_onnx.replace('_onnx.onnx', '.pth'),
                device
            )
            self.text_tower = build_text_tower(
                text_tower_onnx.replace('_onnx.onnx', '.pth'),
                device
            )

    def insert(self, data: list[dict], batch_size=32):
        """
        Insert data into ChromaDB collection.
        
        Each item should have:
        {
            "image_id": unique identifier,
            "image_path": path to image,
            "caption_id": unique identifier,
            "caption": text caption
        }
        """
        if self.use_onnx:
            # Extract image paths and captions
            image_paths = [item['image_path'] for item in data]
            captions = [item['caption'] for item in data]
            
            # Encode in batches
            image_embeddings_list = []
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = [Image.open(p).convert('RGB') for p in batch_paths]
                embeddings = encode_images(self.vision_tower, batch_images)
                image_embeddings_list.append(embeddings)
            
            import numpy as np
            image_embeddings = np.vstack(image_embeddings_list)
            
            # Insert into DB
            from chroma_db import ChromaDB
            db_instance = self.db.db
            
            for idx, item in enumerate(data):
                db_instance.add(
                    ids=[str(item['image_id'])],
                    embeddings=[image_embeddings[idx].tolist()],
                    metadatas=[{
                        'image_path': item['image_path'],
                        'caption': item['caption'],
                        'caption_id': str(item['caption_id'])
                    }]
                )
        else:
            # Use PyTorch insertion (original logic)
            self.db.insert(self.vision_tower, data, batch_size, self.device)

    def search(self, query, top_k=5):
        """
        Perform multimodal search.
        
        Args:
            query: Text string or PIL Image
            top_k: Number of results to return
        
        Returns:
            Search results from ChromaDB
        """
        if isinstance(query, str):
            if self.use_onnx:
                embedding = encode_texts(self.text_tower, query)[0]
            else:
                embedding = text_embedding(self.text_tower, query, self.device)
        elif isinstance(query, Image.Image):
            if self.use_onnx:
                embedding = encode_images(self.vision_tower, query)[0]
            else:
                embedding = image_embedding(self.vision_tower, query, self.device)
        else:
            raise ValueError("Query must be a string or PIL Image.")
        
        results = self.db.search(embedding, top_k)
        return results

    def visualize(self, results):
        """Visualize search results."""
        self.db.result_visualize(results)


# Backward compatibility - keep original class name with PyTorch default
class MultimodalRetrieval(MultimodalRetrievalONNX):
    """
    MultimodalRetrieval with PyTorch by default for backward compatibility.
    Use MultimodalRetrievalONNX for ONNX models.
    """
    
    def __init__(
        self,
        db_path="./clip_db",
        collection_name="clip_retrieval",
        vision_tower_weight="/root/Project/brick_vidgen/vnclip/deploy/weight/vision_tower.pth",
        text_tower_weight="/root/Project/brick_vidgen/vnclip/deploy/weight/text_tower.pth",
        device='cpu',
        use_onnx=False  # Default to PyTorch for backward compatibility
    ):
        """Initialize with PyTorch models by default."""
        super().__init__(
            db_path=db_path,
            collection_name=collection_name,
            vision_tower_onnx=vision_tower_weight,
            text_tower_onnx=text_tower_weight,
            device=device,
            use_onnx=use_onnx
        )
