# VNCLIP (ConvNeXt + PhoBERT)

**Vietnamese Vision–Language Model (CLIP-style)**
Training • Evaluation • Deployment

---

## 📌 Overview

This repository provides a full pipeline for training, evaluating, and deploying a Vietnamese vision–language model inspired by CLIP.

**Architecture**

* **Vision Encoder**: ConvNeXt-Small
* **Text Encoder**: PhoBERT (`vinai/phobert-base-v2`)
* **Training Objective**: CLIP-style contrastive learning with projection heads

<p align="center">
  <img src="./report/VNECLIP.jpg" alt="Model Architecture" width="600"/>
</p>

---

## 📂 Repository Structure

| Component          | Description                                        |
| ------------------ | -------------------------------------------------- |
| `train*.py`        | Training scripts (various stages & configurations) |
| `finetune.py`      | Fine-tuning script                                 |
| `evaluation.ipynb` | Retrieval evaluation (Hit@K)                       |
| `deploy/`          | Demo application (FastAPI + frontend)              |
| `checkpoint.py`    | Checkpoint management                              |
| `wandb_logger.py`  | W&B logging utility                                |

---

## ⚠️ Path Configuration (Important)

Some scripts use **hard-coded absolute paths**, e.g.:

```
/root/Project/brick_vidgen/vnclip/...
```

To run the project:

* ✅ Recommended: keep the same directory structure
* ⚙️ Alternative: update paths manually in scripts

Also check:

```
deploy/backend/config.py
```

---

## 🛠️ Environment Setup

### Option A — Conda (Recommended)

```bash
cd /root/Project/brick_vidgen/vnclip

conda create -n vnclip python=3.12 -y
conda activate vnclip

# Install PyTorch (match your CUDA version)
pip install -r requirements.txt
```

---

### Option B — Virtual Environment

```bash
cd /root/Project/brick_vidgen/vnclip

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

---

### ✅ GPU Check

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## ⚙️ Pretrained Components

### ConvNeXt (Download [here](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth))

Expected path:

```
weights/convnect_small/convnext_small_1k_224_ema.pth
```

### PhoBERT

Downloaded automatically via Hugging Face or [here](https://huggingface.co/vinai/phobert-base-v2).:

```
vinai/phobert-base-v2
```

---

## 📊 Datasets

### 1. COCO-2017-Vietnamese (Parquet)

```
dataset/coco-2017-vietnamese/
  ├── data/
  │   ├── train-*.parquet
  │   └── validation-*.parquet
```

Hugging Face dataset [here](https://huggingface.co/datasets/ai-enthusiasm-community/coco-2017-vietnamese):

```
ai-enthusiasm-community/coco-2017-vietnamese
```

---

### 2. KTVIC Dataset

```
dataset/ktvic_dataset/
  ├── train-images/
  ├── train_data.json
  ├── public-test-images/
  └── test_data.json
```

**Expected JSON schema**

* `images`: `{id, filename}`
* `annotations`: `{id, image_id, caption, segment_caption}`

---

### 3. UITVIC Dataset

```
dataset/uitvic_dataset/
  ├── coco_uitvic_train/
  ├── uitvic_captions_train2017.json
  ├── coco_uitvic_test/
  └── uitvic_captions_test2017.json
```

* COCO-style annotations
* `segment_caption` generated via `underthesea.word_tokenize`

---

## 📈 Experiment Tracking (W&B)

### Recommended

```bash
export WANDB_API_KEY="your_key"
```

### Alternative (used in scripts)

```
/root/Project/brick_vidgen/vnclip/wandb_key.txt
```

---

## 🚀 Training

All training scripts are standalone Python files (no CLI). You may need to adjust:

* Dataset paths
* Batch size
* Learning rate
* Checkpoint directory

---

### 1. Baseline (COCO only)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

* Encoders frozen
* Train projection heads only

---

### 2. Multi-dataset Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_v2.py
```

Datasets:

* COCO + KTVIC + UITVIC

---

### 3. Phase 2 Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_v2_phase2.py
```

* Load checkpoint
* Adjust trainable layers

---

### 4. Full Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_v2_full.py
```

---


## 📊 Evaluation

Run:

```
evaluation.ipynb
```

### Metrics

* Image → Text retrieval (Hit@K)
* Text → Image retrieval (Hit@K)

<p align="center">
  <img src="./report/vneclip_eval.png" alt="Evaluation" width="600"/>
</p>

---

## 🌐 Demo Deployment

Located in:

```
deploy/
```

---

### Features

* Camera input (224×224)
* Zero-shot classification
* FastAPI backend + static frontend

---

### Run Demo

```bash
cd deploy
chmod +x run.sh stop.sh
./run.sh
```

**Access**

* Backend: [http://localhost:5000](http://localhost:5000)
* Frontend: [http://localhost:5001](http://localhost:5001)

Stop:

```bash
./stop.sh
```

---

### Use Your Own Model

Replace:

```
deploy/weight/vision_tower.pth
```

Ensure compatibility with:

```
EncoderTower (in inference.py)
```

If updating prompts:

```
deploy/weight/prompts.txt
```

Then regenerate:

```
prompt_embedding.npy
```

---

## 🛠️ Troubleshooting

| Issue            | Solution                          |
| ---------------- | --------------------------------- |
| Missing packages | `pip install -r requirements.txt` |
| Path errors      | Update hard-coded paths           |
| OOM (GPU/CPU)    | Reduce `BATCH_SIZE`               |
| W&B issues       | Check API key setup               |

---

## 📌 Notes

* Scripts are optimized for internal experimentation → expect manual edits
* Large batch sizes may require high-memory GPUs
* Multi-stage training is recommended for best performance

---
