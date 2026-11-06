"""CLIP text embedding pipeline.

Owner: Member B (embedding and environment)

Job: encode captions into CLIP ViT-L/14 text embeddings, the conditioning
used by Stable Diffusion 1.5.

Stable Diffusion 1.5 conditions the UNet on the last hidden state of the
CLIP ViT-L/14 text encoder: every caption is padded/truncated to 77 tokens
and each token becomes a 768-dim vector, giving a (77, 768) tensor per
caption. We load the tokenizer and text encoder straight from the SD 1.5
checkpoint so the embeddings are exactly what the pipeline will use.

Usage:
    python src/embed.py                      # all captions in data/subset.csv
    python src/embed.py --limit 100          # quick run on the first 100 rows
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "subset.csv"
OUT_PATH = ROOT / "data" / "embeddings.pt"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MAX_LEN = 77      # CLIP context length
EMBED_DIM = 768   # ViT-L/14 text hidden size


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text_encoder(device=None):
    """Load the CLIP ViT-L/14 tokenizer and text encoder used by SD 1.5."""
    device = device or get_device()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    encoder.to(device).eval()
    return tokenizer, encoder


def tokenize(tokenizer, captions):
    return tokenizer(
        list(captions),
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )


@torch.no_grad()
def encode(captions, tokenizer, encoder, batch_size=64, show_progress=False):
    """Encode captions with the CLIP text encoder.

    Returns:
        hidden: (N, 77, 768) last hidden state, the SD 1.5 conditioning.
        pooled: (N, 768) pooled output (end-of-text token), one vector per
            caption, used for caption-to-caption similarity.
    """
    captions = list(captions)
    device = next(encoder.parameters()).device
    hidden, pooled = [], []
    batches = range(0, len(captions), batch_size)
    for i in tqdm(batches, desc="encoding", disable=not show_progress):
        tokens = tokenize(tokenizer, captions[i : i + batch_size]).to(device)
        out = encoder(input_ids=tokens.input_ids)
        hidden.append(out.last_hidden_state.float().cpu())
        pooled.append(out.pooler_output.float().cpu())
    return torch.cat(hidden), torch.cat(pooled)


def cosine_matrix(vectors):
    """Pairwise cosine similarity between rows of an (N, D) tensor."""
    v = F.normalize(vectors, dim=-1)
    return v @ v.T


def save_embeddings(path, df, hidden, pooled):
    torch.save(
        {
            "image_id": torch.tensor(df["image_id"].values),
            "caption": df["caption"].tolist(),
            "hidden": hidden.half(),   # fp16 halves the file size (~240 MB)
            "pooled": pooled.half(),
            "model_id": MODEL_ID,
        },
        path,
    )


def load_embeddings(path=OUT_PATH):
    return torch.load(path, weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    device = get_device()
    print(f"device: {device}")
    tokenizer, encoder = load_text_encoder(device)
    hidden, pooled = encode(df["caption"], tokenizer, encoder, args.batch_size, show_progress=True)

    assert hidden.shape == (len(df), MAX_LEN, EMBED_DIM), hidden.shape
    assert not torch.isnan(hidden).any(), "NaNs in embeddings"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(args.out, df, hidden, pooled)
    print(f"saved {tuple(hidden.shape)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
