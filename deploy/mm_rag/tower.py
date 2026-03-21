import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from transformers import AutoTokenizer
from model import *

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

INFER_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

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
    
def build_vision_tower(weight_path, device='cpu'):
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

def build_text_tower(weight_path, device='cpu'):
    """ Builds the text tower by loading the PhoBERT backbone and projection head with the provided weights. """
    # Init PhoBERT and ProjectionHead
    phobert = PhoBERT(model_name="vinai/phobert-base-v2")
    projection = ProjectionHead(768, 384, 0.1)
    # Init EncoderTower
    text_tower = EncoderTower(phobert, projection, None)
    # Load weights and move to device
    weight = torch.load(weight_path, map_location=device)
    text_tower.load_state_dict(weight)
    text_tower.eval()
    text_tower.to(device) if device else text_tower.cpu()
    return text_tower