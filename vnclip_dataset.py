import os
import torch
import random
from PIL import Image
from underthesea import word_tokenize
from torch.utils.data import Dataset

class VNClipDataset(Dataset):
    def __init__(self, image_imgid2path, caption_imgid2info, tokenizer, transform=None, use_wordseg=False):
        self.image_imgid2path = image_imgid2path
        self.caption_imgid2info = caption_imgid2info
        self.image_keys = list(image_imgid2path.keys())
        self.tokenizer = tokenizer
        self.transform = transform
        self.use_wordseg = use_wordseg

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
        return len(self.image_keys)

    def __getitem__(self, idx):
        image_path = self.image_imgid2path[self.image_keys[idx]]
        caption_infos = self.caption_imgid2info[self.image_keys[idx]]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption_info = random.choice(caption_infos)
        if self.use_wordseg:
            raw_caption = caption_info["caption"]
            seg_caption = self.word_segment(raw_caption)
            tokenized_caption = self.tokenize_caption(seg_caption)
        else:
            raw_caption, seg_caption = caption_info["caption"], caption_info["segment_caption"]
            tokenized_caption = self.tokenize_caption(seg_caption)
        return image, tokenized_caption, raw_caption