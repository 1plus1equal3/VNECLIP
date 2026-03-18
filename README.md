# VNCLIP (ConvNeXt + PhoBERT) — Training, Evaluation, Demo Deploy

This repository trains and evaluates a Vietnamese vision–language model (CLIP-style) and includes a lightweight demo app.

**Core idea**
- Vision encoder: ConvNeXt (small)
- Text encoder: PhoBERT (vinai/phobert-base-v2)
- CLIP-style contrastive learning with projection heads

**What’s in here**
- Training scripts: `train.py`, `train_v2.py`, `train_v2_phase2.py`, `train_v2_full.py`, `finetune.py`
- Evaluation notebook: `evaluation.ipynb`
- Demo app: `deploy/` (FastAPI backend + static frontend)

---

## 0) Quick notes about paths (important)

Several scripts currently use **hard-coded absolute paths** like:
`/root/Project/brick_vidgen/vnclip/...`

To run them without edits you must either:
- keep this repo at that same path, **or**
- edit the paths inside the scripts to match your local clone.

The demo app under `deploy/` also uses absolute paths (see `deploy/backend/config.py`).

---

## 1) Environment setup

### Option A — Conda (recommended)

```bash
cd /root/Project/brick_vidgen/vnclip

conda create -n vnclip python=3.12.* -y
conda activate vnclip

# Install PyTorch first (pick the correct command for your CUDA version)
# See: https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

### Option B — venv

```bash
cd /root/Project/brick_vidgen/vnclip
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### GPU check

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 2) Weights & pretrained components

### ConvNeXt weights

Training scripts expect ConvNeXt weights in:
- `weights/convnect_small/convnext_small_1k_224_ema.pth`

These files already exist in this repo.

### PhoBERT weights

Downloaded automatically by Hugging Face when you run training/eval:
- `vinai/phobert-base-v2`

---

## 3) Datasets: download & expected layouts

This repo uses 3 main datasets (paths are referenced directly in the training scripts).

### 3.1 COCO-2017-Vietnamese (Parquet)

Scripts `train.py`, `train_v2*.py` load Parquet shards from:

```
dataset/coco-2017-vietnamese/
	README.md
	data/
		train-*.parquet
		validation-*.parquet
```

This dataset is also available on Hugging Face:
- `ai-enthusiasm-community/coco-2017-vietnamese`

Download (optional, if you don’t already have the Parquet files locally):

```python
from datasets import load_dataset

ds = load_dataset("ai-enthusiasm-community/coco-2017-vietnamese")
print(ds)
```

If you want to materialize Parquet shards to match the training scripts, you can use Hugging Face Datasets export utilities (requires `pyarrow`).

### 3.2 KTVIC dataset

Training scripts reference:

```
dataset/ktvic_dataset/
	train-images/
	train_data.json
	public-test-images/
	test_data.json
```

The JSON schema expected by `train_v2*.py` / `finetune.py`:
- `images`: list with `id`, `filename`
- `annotations`: list with `id`, `image_id`, `caption`, `segment_caption`

### 3.3 UITVIC dataset

Training scripts reference:

```
dataset/uitvic_dataset/
	coco_uitvic_train/
		coco_uitvic_train/   # image files
	uitvic_captions_train2017.json
	coco_uitvic_test/
	uitvic_captions_test2017.json
```

The UITVIC annotations are COCO-style (`images` + `annotations`). The scripts build `segment_caption` on the fly with `underthesea.word_tokenize`.

---

## 4) W&B (Weights & Biases) logging

Training scripts use `WandbLogger` in `wandb_logger.py`.

### Recommended: use an environment variable

```bash
export WANDB_API_KEY="<your_key_here>"
```

### Current scripts: also read from a local file

Some scripts do:
`open("/root/Project/brick_vidgen/vnclip/wandb_key.txt").read()`

So make sure you have a file at that path containing your key (or edit the scripts to read from `WANDB_API_KEY`).

---

## 5) Start training

All training scripts are simple Python files (no CLI flags). You typically edit:
- dataset paths
- batch size / workers
- checkpoint output directory
- learning rate and freeze/unfreeze logic

### 5.1 Baseline training (COCO-2017-Vietnamese only)

Runs `train.py` (freezes vision + text encoders, trains projection heads):

```bash
cd /root/Project/brick_vidgen/vnclip
CUDA_VISIBLE_DEVICES=0 python train.py
```

The helper script `train.sh` currently runs:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 5.2 Multi-dataset training (COCO + KTVIC + UITVIC)

Runs `train_v2.py`:

```bash
cd /root/Project/brick_vidgen/vnclip
CUDA_VISIBLE_DEVICES=0 python train_v2.py
```

### 5.3 Phase 2 training (example)

Runs `train_v2_phase2.py` (loads a checkpoint and changes what’s trainable; see the file):

```bash
cd /root/Project/brick_vidgen/vnclip
CUDA_VISIBLE_DEVICES=0 python train_v2_phase2.py
```

### 5.4 “Full” training (example)

Runs `train_v2_full.py`:

```bash
cd /root/Project/brick_vidgen/vnclip
CUDA_VISIBLE_DEVICES=0 python train_v2_full.py
```

---

## 6) Checkpoints (save/resume)

Checkpointing is handled by `CheckpointManager` in `checkpoint.py`.

Each training script sets its own `save_dir`, e.g.:
- `checkpoints_v3_160326/`
- `checkpoints_v3_160326_phase2/`

Files are written as:
`checkpoint_epoch_<EPOCH>_<YYYY-MM-DD>.pth`

To resume, edit the relevant script and uncomment / adjust its `ckpt_path` logic.

---

## 7) Evaluation

Evaluation is primarily done via the notebook `evaluation.ipynb`.

### 7.1 Evaluate VNCLIP retrieval (recommended path)

1. Open `evaluation.ipynb`.
2. In the “Model Initialization” section, set:
	 - `ckpt_path` to the checkpoint you want to evaluate
3. In the dataset preparation section, pick KTVIC or UITVIC test paths.
4. Run the cells to compute similarity matrices and retrieval Hit@K.

The notebook computes:
- Image-to-text retrieval Hit@K
- Text-to-image retrieval Hit@K

### 7.2 Optional: evaluate baselines (Original CLIP / Multilingual CLIP)

The notebook also contains baseline evaluation code that uses extra dependencies such as `clip` and `multilingual_clip` and expects the OpenAI CLIP weight file under:
- `weights/original_clip/ViT-L-14.pt`

Install extras only if you need that section.

---

## 8) Deploy demo (FastAPI backend + static frontend)

The demo lives under `deploy/`.

### 8.1 What the demo does

- Frontend opens the camera, captures a 224×224 image, and sends it to the backend.
- Backend runs **zero-shot classification** by encoding the image with a “vision tower” and comparing to precomputed prompt embeddings.

Key files:
- Backend API: `deploy/backend/main.py`
- Inference helpers: `deploy/backend/inference.py`
- Paths to weights/prompts: `deploy/backend/config.py`
- Weights/prompts: `deploy/weight/`

### 8.2 Run demo

```bash
cd /root/Project/brick_vidgen/vnclip/deploy
chmod +x run.sh stop.sh
./run.sh
```

By default:
- Backend: http://localhost:5000
- Frontend: http://localhost:5001

Stop:

```bash
./stop.sh
```

### 8.3 Swap in your own trained model

The demo backend loads `deploy/weight/vision_tower.pth`. This is expected to match the `EncoderTower` defined in `deploy/backend/inference.py` (ConvNeXt-small + `ProjectionHead`).

To update the demo with your new checkpoint, you can export the vision tower state dict from your trained VNCLIP model (see the commented “save tower” code in `evaluation.ipynb`) and replace:
- `deploy/weight/vision_tower.pth`

If you change the prompt set (`deploy/weight/prompts.txt`), you must regenerate:
- `deploy/weight/prompt_embedding.npy`

---

## 9) Troubleshooting

- **Missing packages**: if training errors mention `wandb` or `torchmetrics`, re-run `pip install -r requirements.txt`.
- **Path errors**: update hard-coded paths inside the scripts (or clone into `/root/Project/brick_vidgen/vnclip`).
- **CPU RAM / batch size**: `train.py` uses very large batch sizes by default; reduce `BATCH_SIZE` if you OOM.
