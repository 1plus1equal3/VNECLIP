import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from model import *
from config import *

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

class EncoderTower(nn.Module):
    def __init__(self, encoder, projection, logit_scale=None):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.logit_scale = logit_scale

    def forward(self, images):
        features = self.encoder(images)
        projected = self.projection(features)
        return projected

def build_vision_tower(weight_path=VISION_TOWER_WEIGHT_PATH, device='cpu'):
    """ Builds the vision tower by loading the ConvNeXt backbone and projection head with the provided weights. """
    # Init ConvNeXt and ProjectionHead
    convnext = convnext_small(pretrained=False, weight_path=None)
    projection = ProjectionHead(768, 384, 0.1)
    # Init EncoderTower
    vision_tower = EncoderTower(convnext, projection, nn.Parameter(torch.ones([]) * np.log(1 / 0.07)))
    # Load weights and move to device
    weight = torch.load(weight_path, map_location=device)
    vision_tower.load_state_dict(weight)
    vision_tower.eval()
    vision_tower.to(device) if device else vision_tower.cpu()
    return vision_tower

def encode_image(model: EncoderTower, image: Image.Image):
    """ Encodes the input image into a feature embedding using the vision tower. """
    image = image.convert("RGB")
    image_tensor = INFER_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        image_embedding = model(image_tensor)
        image_embedding = F.normalize(image_embedding, dim=-1)
    return image_embedding.cpu().numpy()

def zero_shot_classify(image, vision_tower, prompt_embeddings=PROMPT_EMBEDDINGS, prompts=PROMPTS, top_k=3):
    image_embedding = encode_image(vision_tower, image)
    logit_scale = vision_tower.logit_scale.exp().item()
    similarities = logit_scale * image_embedding @ prompt_embeddings.T
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    retrieved_prompts = [prompts[idx] for idx in top_indices]
    return retrieved_prompts