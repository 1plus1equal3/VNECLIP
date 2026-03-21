import os
import random
import chromadb
from retrieval_onnx import MultimodalRetrievalONNX
from PIL import Image
import json

def load_json(file_path):
    """ Load JSON data from a file. """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, file_path):
    """ Save JSON data to a file. """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Initialize with ONNX models for better edge device performance
retrieval_client = MultimodalRetrievalONNX(
    db_path="./clip_db",
    collection_name="clip_retrieval",
    vision_tower_onnx="../weight/vision_tower.onnx",
    text_tower_onnx="../weight/text_tower.onnx",
    device="cpu",
    use_onnx=True
)

# Test simple search
## Test text query
# text_query = "Một con mèo đang nằm trên bàn."
# results = retrieval_client.search(text_query, top_k=5)
# print("Text Query Results:")
# retrieval_client.visualize(results)
## Test image query
image_query = Image.open("/root/Project/brick_vidgen/vnclip/deploy/sample/cat.jpg").convert('RGB')
results = retrieval_client.search(image_query, top_k=5)
print("Image Query Results:")
retrieval_client.visualize(results)
