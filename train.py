import os
import sys
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchmetrics import MeanMetric
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer
from clip_dataset import CLIPDataset
from model import *
from checkpoint import CheckpointManager
from wandb_logger import WandbLogger
from metric import *

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Number of available cpu: {os.cpu_count()}')

# Process dataset files
train_data_files = glob.glob("/root/Project/brick_vidgen/vnclip/dataset/coco-2017-vietnamese/data/train*.parquet")
train_data_files = sorted(train_data_files)  # Ensure consistent order
val_data_files = glob.glob("/root/Project/brick_vidgen/vnclip/dataset/coco-2017-vietnamese/data/val*.parquet")
val_data_files = sorted(val_data_files)  # Ensure consistent order
data_files = train_data_files + val_data_files
print(f"Found {len(data_files)} data files.")

train_hf_dataset = load_dataset(
    "parquet",
    data_files=train_data_files,
    num_proc=8,
)
train_raw_dataset = train_hf_dataset['train']

val_hf_dataset = load_dataset(
    "parquet",
    data_files=val_data_files,
)
val_raw_dataset = val_hf_dataset['train']

# image_root_dir = "/root/Project/brick_vidgen/vnclip/dataset/ktvic_dataset/public-test-images"
# dataset_info = load_json("/root/Project/brick_vidgen/vnclip/dataset/ktvic_dataset/test_data.json")
# images, annotations = dataset_info['images'], dataset_info['annotations']
# test_raw_dataset = []
# img_ids = set()
# for ann in annotations:
#     img_id = ann['image_id']
#     if img_id not in img_ids:
#         img_ids.add(img_id)
#         img_path = os.path.join(image_root_dir, f"{img_id:011d}.jpg")
#         if os.path.exists(img_path):
#             # Verify that the image can be opened
#             try:
#                 with Image.open(img_path).convert('RGB') as img:
#                     img.verify()  # Verify that it's a valid image
#             except (IOError, SyntaxError) as e:
#                 print(f"Warning: Skipping corrupted image {img_path} - {e}")
#                 continue
#             test_raw_dataset.append({
#                 "image": Image.open(img_path),
#                 "caption_vi": ann['segment_caption']
#             })

# Define transformations for training and validation sets
train_transform = T.Compose([
    T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

val_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# Initialize datasets for training and validation
train_dataset = CLIPDataset(train_raw_dataset, tokenizer, transform=train_transform)
val_dataset = CLIPDataset(val_raw_dataset, tokenizer, transform=val_transform)
# test_dataset = CLIPDataset(test_raw_dataset, tokenizer, transform=val_transform)

# Build dataloaders for training and validation
BATCH_SIZE = 192
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# Initialize model components
## Initialize vision encoder (ConvNeXt)
cfg = {
    "pretrained": True,
    "weight_path": "/root/Project/brick_vidgen/vnclip/weights/convnect_small/convnext_small_1k_224_ema.pth"
}
convnext = convnext_small(**cfg)
print(f"Model architecture:\n{convnext}")
total, trainable = count_params(convnext)
print(f"Total parameters: {total}")
print(f"Trainable parameters: {trainable}")
## Initialize text encoder (PhoBERT)
phobert = PhoBERT("vinai/phobert-base-v2")
print(f"Model architecture:\n{phobert}")
total, trainable = count_params(phobert)
print(f"Total parameters: {total}")
print(f"Trainable parameters: {trainable}")
## Initialize VNECLIP model
prj_cfg = {
    "input_dim": 768,
    # "hidden_dim": 384,
    "projection_dim": 384,
    "dropout": 0.1
}
cfg = {
    "vision_encoder": convnext,
    "text_encoder": phobert,
    "vision_prj_cfg": prj_cfg,
    "text_prj_cfg": prj_cfg
}
vnclip = VNECLIP(**cfg).to(DEVICE)
print(f"Model architecture:\n{vnclip}")
total, trainable = count_params(vnclip)
print(f"Total parameters: {total}")
print(f"Trainable parameters: {trainable}")

# Training config
def set_gradient_state(model, requires_grad):
    """ Utility function to set the requires_grad attribute for all parameters in a model. """
    for param in model.parameters():
        param.requires_grad = requires_grad

# #! TEMPORARY FREEZE VISION ENCODER and TEXT ENCODER
set_gradient_state(vnclip.vision_encoder, True)
set_gradient_state(vnclip.text_encoder, True)

# Initialize optimizer (AdamW)
optimizer = optim.AdamW(vnclip.parameters(), lr=1e-4)

# Initialize metrics
training_metrics = {
    "loss_img": MeanMetric().to(DEVICE),
    "loss_txt": MeanMetric().to(DEVICE)
}
eval_metrics = deepcopy(training_metrics)

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(save_dir='/root/Project/brick_vidgen/vnclip/checkpoints_v2', max_checkpoints=10)
start_epoch = 1  # Default start epoch if no checkpoint is found
try:
    ckpt_path = "/root/Project/brick_vidgen/vnclip/checkpoints_v2/checkpoint_epoch_10_2026-03-14.pth"
    start_epoch, _ = checkpoint_manager.load_checkpoint(vnclip, optimizer, ckpt_path)
    print(f"Resuming training from epoch {start_epoch}")
except IndexError:
    print("No checkpoint found, starting training from scratch.")

# optimizer = optim.AdamW(vnclip.parameters(), lr=5e-5)

# Initialize Weights & Biases logger
wandb_logger = WandbLogger(project_name="VNECLIP", api_key=open("/root/Project/brick_vidgen/vnclip/wandb_key.txt").read())  # Replace with your actual W&B API key

def train_step(batch):
    """ A training step for one epoch """ 
    # Parsing batch data
    images, encoded_inputs, _ = batch
    images = images.to(DEVICE)
    for k in encoded_inputs:
        encoded_inputs[k] = encoded_inputs[k].to(DEVICE)
    # Forward pass and loss computation
    optimizer.zero_grad()
    image_loss, text_loss = vnclip(images, encoded_inputs)
    loss = (image_loss + text_loss) / 2.0
    loss.backward()
    optimizer.step()
    return image_loss.detach(), text_loss.detach()

def eval_step(batch):
    """ An evaluation step for one epoch """
    with torch.no_grad():
        images, encoded_inputs, _ = batch
        images = images.to(DEVICE)
        for k in encoded_inputs:
            encoded_inputs[k] = encoded_inputs[k].to(DEVICE)
        image_loss, text_loss = vnclip(images, encoded_inputs)
    return image_loss.detach(), text_loss.detach()

def step_logging(step, total_steps, metrics):
    """ Logs metrics at each step. """
    sys.stdout.write(f"\rStep {step}/{total_steps}: " + 
        " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])) 

def log_text2img_retrieval(epoch, batch, model, num_queries=5, top_k=5):
    """ Logs text-to-image retrieval results for a batch of data. """
    def denormalize(img):
        """ Utility function to denormalize an image tensor for visualization. """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = img * std + mean
        return torch.clamp(img, 0, 1)
    # Parsing batch data
    images, encoded_inputs, captions = batch
    images = images.to(DEVICE)
    for k in encoded_inputs:
        encoded_inputs[k] = encoded_inputs[k].to(DEVICE)
    vnclip.eval()
    with torch.no_grad():
        similarities = model.predict(images, encoded_inputs)
    similarities = similarities.T.cpu()
    rand_idx = random.sample(range(len(captions)), num_queries)
    fig, ax = plt.subplots(num_queries, top_k, figsize=(5*top_k, 5*num_queries))
    # handle case num_queries = 1
    if num_queries == 1:
        ax = [ax]
    for row, idx in enumerate(rand_idx):
        sim_scores = similarities[idx]
        top_indices = torch.topk(sim_scores, top_k).indices
        caption = captions[idx]
        caption_idx = top_k // 2
        for col, img_idx in enumerate(top_indices):
            img = denormalize(images[img_idx].cpu()).permute(1, 2, 0)
            ax[row][col].imshow(img)
            ax[row][col].axis("off")
            score = sim_scores[img_idx].item()
            title = f"\nRank {col+1} - Sim {score:.3f}"
            if caption_idx == col:
                title = f"Caption: {caption}" + title
            ax[row][col].set_title(title)
    plt.tight_layout()
    wandb_logger.log_image(fig, caption=f"Text-to-Image Retrieval Visualization Epoch {epoch}")
    plt.show()


def main(epochs, start_epoch=1, ckpt_epoch=1, viz_epoch=1, eval_epoch=2):
    """ Main training loop for the VNECLIP model. """
    for epoch in range(start_epoch, epochs+1):
        try:
            # Training phase
            vnclip.train()
            reset_metrics(training_metrics)
            total_steps = len(train_loader)
            print(f"\nEpoch {epoch}/{epochs} - Training:")
            for step, batch in enumerate(train_loader, 1):
                image_loss, text_loss = train_step(batch)
                metrics = {"loss_img": image_loss, "loss_txt": text_loss}
                update_metrics(training_metrics, metrics)
                step_logging(step, total_steps, metrics)
                wandb_logger.log_metrics({f"train_{k}": v.item() for k, v in metrics.items()})
            # Checkpointing
            if epoch % ckpt_epoch == 0:
                metrics = compute_metrics(training_metrics)
                print()
                checkpoint_manager.save_checkpoint(vnclip, optimizer, epoch, metrics)
            # Visualize text-to-image retrieval results for a random batch from the validation set
            if epoch % viz_epoch == 0:
                batch = next(iter(val_loader))
                log_text2img_retrieval(epoch, batch, vnclip, num_queries=5, top_k=5)
            # Evaluation phase
            if epoch % eval_epoch == 0:
                vnclip.eval()
                reset_metrics(eval_metrics)
                with torch.no_grad():
                    for batch in val_loader:
                        image_loss, text_loss = eval_step(batch)
                        update_metrics(eval_metrics, {"loss_img": image_loss, "loss_txt": text_loss})
                eval_results = compute_metrics(eval_metrics)
                wandb_logger.log_metrics({f"eval_{k}": v for k, v in eval_results.items()}, step=epoch)
                print(f"\nEpoch {epoch} Evaluation Metrics: {eval_results}\n")
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current checkpoint...")
            checkpoint_manager.save_checkpoint(vnclip, optimizer, epoch, compute_metrics(training_metrics))
            print("Checkpoint saved. Exiting.")
            torch.cuda.empty_cache()
            break
    wandb_logger.finish()


train_cfg = {
    "epochs": 100,
    "start_epoch": start_epoch,
    "ckpt_epoch": 1,
    "viz_epoch": 1,
    "eval_epoch": 2,
}

main(**train_cfg)