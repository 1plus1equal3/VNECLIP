import os
import torch
from underthesea import word_tokenize
from torch.utils.data import Dataset

class CLIPDataset(Dataset):
    def __init__(self, dataset, tokenizer, transform=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def word_segment(self, text):
        """ Perform word segmentation using VnCoreNLP """
        return word_tokenize(text, format='text')

    def tokenize_caption(self, caption):
        encoded = self.tokenizer(
            caption,
            max_length=50,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }

    def __len__(self):
        """ Return the total number of samples in the dataset """
        return len(self.dataset)

    def __getitem__(self, idx):
        """ Retrieve the image and caption for a given index, apply transformations, and return them as tensors """
        sample = self.dataset[idx]
        image = sample['image']
        image = image.convert("RGB")
        caption = sample['caption_vi']
        if self.transform:
            image = self.transform(image) # Apply transformations (e.g., normalization, tensor conversion)
        encoded_input = self.tokenize_caption(self.word_segment(caption))
        return image, encoded_input, caption