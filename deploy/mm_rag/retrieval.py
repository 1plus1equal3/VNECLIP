import os
from PIL import Image
from chroma_db import ImageVecDB
from embedder import *

class MultimodalRetrieval:
    """ Multimodal Retrieval and Generation (MM-RAG) system for image retrieval based on text or image queries."""
    def __init__(
        self,
        # chroma_db settings
        db_path="./clip_db",
        collection_name="clip_retrieval",
        # model settings
        vision_tower_weight="/root/Project/brick_vidgen/vnclip/deploy/weight/vision_tower.pth",
        text_tower_weight="/root/Project/brick_vidgen/vnclip/deploy/weight/text_tower.pth",
        device='cpu'
    ):
        self.db = ImageVecDB(db_path, collection_name)
        self.vision_tower = build_vision_tower(vision_tower_weight, device)
        self.text_tower = build_text_tower(text_tower_weight, device)
        self.device = device

    def insert(self, data: list[dict], batch_size=32, device='cpu'):
        """ Insert data into the ChromaDB collection. 
            Each item should be a dict with format:
            {
                "image_id": unique identifier for the image,
                "image_path": path to the image file,
                "caption_id": unique identifier for the caption,
                "caption": text caption describing the image
            }
        """
        self.db.insert(self.vision_tower, data, batch_size, device)

    def search(self, query, top_k=5, device='cpu'):
        """ Perform multimodal search and retrieval.
            Query can be either a text string or a PIL Image. 
            Returns the search results from ChromaDB.
        """
        if isinstance(query, str):
            embedding = text_embedding(self.text_tower, query, device)
        elif isinstance(query, Image.Image):
            embedding = image_embedding(self.vision_tower, query, device)
        else:
            raise ValueError("Query must be a string or a PIL Image.")
        results = self.db.search(embedding, top_k)
        return results

    def visualize(self, results):
        """ Visualize the search results."""
        self.db.result_visualize(results)

# Simple Test
if __name__ == "__main__":
    # Initialize retrieval system
    retrieval_client = MultimodalRetrieval(
        db_path="./clip_db",
        collection_name="clip_retrieval",
        vision_tower_weight="/root/Project/brick_vidgen/vnclip/deploy/weight/vision_tower.pth",
        text_tower_weight="/root/Project/brick_vidgen/vnclip/deploy/weight/text_tower.pth",
        device="cpu"
    )
    # Test text query
    text_query = "Một con mèo đang nằm trên bàn."
    results = retrieval_client.search(text_query, top_k=5)
    print("Text Query Results:")
    retrieval_client.visualize(results)
    # Test image query
    image_query = Image.open("/root/Project/brick_vidgen/vnclip/deploy/data/cat.jpg").convert('RGB')
    results = retrieval_client.search(image_query, top_k=5)
    print("Image Query Results:")
    retrieval_client.visualize(results)