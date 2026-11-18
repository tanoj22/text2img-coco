"""CLIP scores for generated images and for the real COCO baseline.

The scorer is torchmetrics.multimodal.CLIPScore with
openai/clip-vit-base-patch16. It is loaded once and placed on CUDA when
that device is available. Scores are 100 times the cosine similarity,
clamped at 0, so they fall between 0 and 100.

Usage (from the repo root):
    python src/evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore

try:
    from logger import COLUMNS, LOG_PATH
except ImportError:
    from .logger import COLUMNS, LOG_PATH

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/clip-vit-base-patch16"
REAL_SCORES_PATH = ROOT / "outputs" / "real_clip_scores.csv"
PROMPT_COLUMNS = ["prompt_id", "image_id", "prompt", "category", "real_image_path"]

_metric = None
_device = None


def _get_metric() -> CLIPScore:
    """Load CLIPScore once. Later calls reuse the same module."""
    global _metric, _device
    if _metric is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _metric = CLIPScore(model_name_or_path=MODEL_NAME).to(_device)
        _metric.eval()
    return _metric


def _image_tensor(image: Image.Image) -> torch.Tensor:
    """RGB uint8 tensor shaped (C, H, W), which CLIPScore expects."""
    rgb = image.convert("RGB")
    array = np.array(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    from_cwd = Path.cwd() / candidate
    if from_cwd.is_file():
        return from_cwd
    return ROOT / candidate


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "none"
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clip_score(image: Image.Image, prompt: str) -> float:
    """Score one image against one prompt. Higher means a closer match."""
    metric = _get_metric()
    metric.reset()
    tensor = _image_tensor(image).to(_device)
    # forward() would run CLIP twice because CLIPScore sets full_state_update.
    with torch.no_grad():
        metric.update(tensor, prompt)
        value = metric.compute()
    metric.reset()
    return float(value.detach().cpu())


def score_log(log_path: str = LOG_PATH) -> pd.DataFrame:
    """Fill clip_score for every log row that does not have one yet."""
    path = _resolve(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"generation log not found: {path}")

    df = pd.read_csv(path)
    if list(df.columns) != COLUMNS:
        raise ValueError(f"log columns must be {COLUMNS}, found {list(df.columns)}")

    for index, row in df.iterrows():
        if not _is_blank(row["clip_score"]):
            continue
        image_path = _resolve(str(row["image_path"]))
        with Image.open(image_path) as image:
            df.at[index, "clip_score"] = clip_score(image, str(row["prompt"]))

    df.to_csv(path, index=False)
    return df


def real_image_baseline(prompts_csv: str = "data/prompts.csv") -> pd.DataFrame:
    """Score each real COCO image against its caption and save the CSV."""
    path = _resolve(prompts_csv)
    prompts = pd.read_csv(path)
    missing = [col for col in PROMPT_COLUMNS if col not in prompts.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")

    rows = []
    for _, row in prompts.iterrows():
        image_path = _resolve(str(row["real_image_path"]))
        with Image.open(image_path) as image:
            score = clip_score(image, str(row["prompt"]))
        rows.append({
            "prompt_id": row["prompt_id"],
            "image_id": row["image_id"],
            "prompt": row["prompt"],
            "category": row["category"],
            "real_image_path": row["real_image_path"],
            "clip_score": score,
        })

    scores = pd.DataFrame(rows)
    REAL_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(REAL_SCORES_PATH, index=False)
    print(f"wrote {len(scores)} scores to {REAL_SCORES_PATH}")
    return scores


def main():
    scores = real_image_baseline()
    mean_score = float(scores["clip_score"].mean())
    print(f"mean CLIP score: {mean_score:.4f}")


if __name__ == "__main__":
    main()
