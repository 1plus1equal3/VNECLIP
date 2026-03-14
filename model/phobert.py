import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERT(nn.Module):
    def __init__(self, model_name="vinai/phobert-base-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, encoded_inputs):
        input_ids = encoded_inputs['input_ids']
        attention_masks = encoded_inputs['attention_mask']
        features = self.model(input_ids, attention_mask=attention_masks)
        return features[0][:, 0, :] # SHAPE: (batch_size, 768)
