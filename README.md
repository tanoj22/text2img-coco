# Text-to-Image Generation on MS COCO

**Goal:** generate images from captions using Stable Diffusion conditioned on CLIP text embeddings.

## Dataset

MS COCO 2017 val split, 2,000 image-caption pairs, 1 caption per image, sampled with category stratification across 80 classes (seed 42).

## Model

Stable Diffusion 1.5 with the CLIP ViT-L/14 text encoder.

## Team Roles

| Member | GitHub username | Owns | Files |
|--------|-----------------|------|-------|
| A (name TBD) | @username-a | Data: download, stratified sampling, cleaning, resize, subset.csv | `src/data.py` |
| B (name TBD) | @username-b | Embedding and environment | `src/embed.py`, `notebooks/baseline_embedding.ipynb`, `requirements.txt` |
| C (Raymond Wang) | @rwang1237 | Tests | `notebooks/tests.ipynb` |
| D (name TBD) | @username-d | Generation and report | `src/generate.py`, `report/` |

## Setup (Anaconda)

```bash
conda create -n dl-final python=3.10 -y
conda activate dl-final
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install PyTorch with the command above **before** `pip install -r requirements.txt`. Otherwise pip installs the CPU-only build to satisfy `torch>=2.1`. On Mac or a CPU-only machine, use `pip install torch torchvision` instead.

## Data Download

Download:

- http://images.cocodataset.org/zips/val2017.zip
- http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Unzip both into `data/`.

## Git Workflow

- Each member works on their own branch named `yourname-<role>`: `tanoj-data` (A), `yourname-embed` (B), `yourname-tests` (C), `yourname-generate` (D).
- Open a pull request into `main` when your work is ready.
- Commit from your own GitHub account.
- Never commit raw data or model weights.
