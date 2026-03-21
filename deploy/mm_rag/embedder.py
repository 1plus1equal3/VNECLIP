import torch
import torch.nn.functional as F
from PIL import Image
from underthesea import word_tokenize
from tower import *

def text_embedding(model: EncoderTower, texts: list[str], device='cpu'):
    """ Get text embedding from the text encoder."""
    # Preprocess text
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        raise ValueError("Input must be a string or a list of strings.")
    seg_texts = [word_tokenize(text, format="text") for text in texts]
    input_ids = []
    attention_masks = []
    for text in seg_texts:
        input = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=50)
        input_ids.append(input["input_ids"]) 
        attention_masks.append(input["attention_mask"]) 
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_mask = torch.cat(attention_masks, dim=0).to(device)
    # Get text embedding
    with torch.no_grad():
        embeddings = model({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        embeddings = F.normalize(embeddings, dim=-1)  # Normalize the embedding
    return embeddings.cpu().numpy()


def image_embedding(model: EncoderTower, images: list[Image.Image], device='cpu'):
    """ Get image embedding from the vision encoder."""
    if isinstance(images, Image.Image):
        images = [images]
    elif not isinstance(images, list):
        raise ValueError("Input must be a PIL Image or a list of PIL Images.")
    # Preprocess image
    images = [INFER_TRANSFORM(image).unsqueeze(0) for image in images]
    images = torch.cat(images, dim=0).to(device)
    # Get image embedding
    with torch.no_grad():
        embeddings = model(images)
        embeddings = F.normalize(embeddings, dim=-1)  # Normalize the embedding
    return embeddings.cpu().numpy()

# Simple Test
if __name__ == "__main__":
    # Build towers
    vision_tower = build_vision_tower("/root/Project/brick_vidgen/vnclip/deploy/weight/vision_tower.pth")
    text_tower = build_text_tower("/root/Project/brick_vidgen/vnclip/deploy/weight/text_tower.pth")
    # Test text embedding
    texts = ["Một con mèo đang nằm trên bàn.", "Một chiếc xe đang chạy trên đường."]
    embeddings = text_embedding(text_tower, texts)
    print("Text embedding shape:", embeddings.shape) # Should be (2, 384)
    # Test image embedding
    image_paths = [
        "/root/Project/brick_vidgen/vnclip/deploy/data/cat.jpg",
        "/root/Project/brick_vidgen/vnclip/deploy/data/chair.jpg"
    ]
    images = [Image.open(path).convert('RGB') for path in image_paths]
    embeddings = image_embedding(vision_tower, images)
    print("Image embedding shape:", embeddings.shape) # Should be (2, 384)