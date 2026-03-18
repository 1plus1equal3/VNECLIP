import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .convnext import ConvNeXt
from .phobert import PhoBERT
from .utils import *

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, projection_dim=192, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, projection_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        _x = self.fc1(x)
        x = self.fc2(self.activation(_x))
        x = self.dropout(x) + _x
        x = self.layer_norm(x)
        return x

class VNECLIP(nn.Module):
    """ VNECLIP model using ConvNeXt and PhoBERT with projection heads for both modalities. """
    def __init__(self, vision_encoder: ConvNeXt, text_encoder: PhoBERT, vision_prj_cfg: dict, text_prj_cfg: dict):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_projection = ProjectionHead(**vision_prj_cfg)
        self.text_projection = ProjectionHead(**text_prj_cfg)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, images):
        features = self.vision_encoder(images)
        projected = self.vision_projection(features)
        return projected

    def encode_text(self, sentences):
        features = self.text_encoder(sentences)
        projected = self.text_projection(features)
        return projected
    
    def predict(self, images, sentences):
        with torch.no_grad():
            image_features = self.encode_image(images)
            text_features = self.encode_text(sentences)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarity = image_features @ text_features.T
        return similarity

    # def forward(self, images, sentences):
    #     image_features = self.encode_image(images)
    #     text_features = self.encode_text(sentences)

    #     image_features = F.normalize(image_features, dim=-1)
    #     text_features = F.normalize(text_features, dim=-1)

    #     # logit_scale = self.logit_scale.exp()
    #     logits = image_features @ text_features.T    
    #     image_similarity = image_features @ image_features.T
    #     text_similarity = text_features @ text_features.T
    #     target = F.softmax((image_similarity + text_similarity) / 2, dim=-1)

    #     image_loss = cross_entropy(logits, target, reduction='mean')
    #     text_loss = cross_entropy(logits.T, target.T, reduction='mean')
    #     return image_loss, text_loss

    def forward(self, images, sentences):
        image_features = self.encode_image(images)
        text_features = self.encode_text(sentences)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_per_image = image_features @ text_features.T   
        logits_per_text = logits_per_image.T

        ground_trunth = torch.arange(len(images)).to(logits_per_image.device)
        loss_image = F.cross_entropy(logits_per_image, ground_trunth)
        loss_text = F.cross_entropy(logits_per_text, ground_trunth)
        return loss_image, loss_text
        

class VNECLIP_v1(nn.Module):
    """ VNECLIP model using ConvNeXt and PhoBERT with projection heads for both modalities. """
    def __init__(self, vision_encoder: ConvNeXt, text_encoder: PhoBERT, vision_prj_cfg: dict, text_prj_cfg: dict):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_projection = ProjectionHead(**vision_prj_cfg)
        self.text_projection = ProjectionHead(**text_prj_cfg)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, images):
        features = self.vision_encoder(images)
        projected = self.vision_projection(features)
        return projected

    def encode_text(self, sentences):
        features = self.text_encoder(sentences)
        projected = self.text_projection(features)
        return projected
    
    def predict(self, images, sentences):
        with torch.no_grad():
            image_features = self.encode_image(images)
            text_features = self.encode_text(sentences)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarity = image_features @ text_features.T
        return similarity

    # def forward(self, images, sentences):
    #     image_features = self.encode_image(images)
    #     text_features = self.encode_text(sentences)

    #     image_features = F.normalize(image_features, dim=-1)
    #     text_features = F.normalize(text_features, dim=-1)

    #     # logit_scale = self.logit_scale.exp()
    #     logits = image_features @ text_features.T    
    #     image_similarity = image_features @ image_features.T
    #     text_similarity = text_features @ text_features.T
    #     target = F.softmax((image_similarity + text_similarity) / 2, dim=-1)

    #     image_loss = cross_entropy(logits, target, reduction='mean')
    #     text_loss = cross_entropy(logits.T, target.T, reduction='mean')
    #     return image_loss, text_loss

    def forward(self, images, sentences):
        image_features = self.encode_image(images)
        text_features = self.encode_text(sentences)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T   
        logits_per_text = logits_per_image.T

        ground_trunth = torch.arange(len(images)).to(logits_per_image.device)
        loss_image = F.cross_entropy(logits_per_image, ground_trunth)
        loss_text = F.cross_entropy(logits_per_text, ground_trunth)
        loss = (loss_image + loss_text) / 2
        return loss

class VNECLIP_v2(nn.Module):
    """ VNECLIP model using ConvNeXt and PhoBERT with projection heads for both modalities. """
    def __init__(self, vision_encoder: ConvNeXt, text_encoder: PhoBERT, vision_prj_cfg: dict, text_prj_cfg: dict):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_projection = ProjectionHead(**vision_prj_cfg)
        self.text_projection = ProjectionHead(**text_prj_cfg)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def encode_image(self, images):
        features = self.vision_encoder(images)
        projected = self.vision_projection(features)
        return projected

    def encode_text(self, sentences):
        features = self.text_encoder(sentences)
        projected = self.text_projection(features)
        return projected
    
    def predict(self, images, sentences):
        with torch.no_grad():
            image_features = self.encode_image(images)
            text_features = self.encode_text(sentences)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarity = image_features @ text_features.T
        return similarity

    # def forward(self, images, sentences):
    #     image_features = self.encode_image(images)
    #     text_features = self.encode_text(sentences)

    #     image_features = F.normalize(image_features, dim=-1)
    #     text_features = F.normalize(text_features, dim=-1)

    #     # logit_scale = self.logit_scale.exp()
    #     logits = image_features @ text_features.T    
    #     image_similarity = image_features @ image_features.T
    #     text_similarity = text_features @ text_features.T
    #     target = F.softmax((image_similarity + text_similarity) / 2, dim=-1)

    #     image_loss = cross_entropy(logits, target, reduction='mean')
    #     text_loss = cross_entropy(logits.T, target.T, reduction='mean')
    #     return image_loss, text_loss

    def forward(self, images, sentences):
        image_features = self.encode_image(images)
        text_features = self.encode_text(sentences)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T + self.logit_bias  
        logits_per_text = logits_per_image.T

        eye = torch.eye(len(images)).to(logits_per_image.device)
        m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
        loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        return loss
