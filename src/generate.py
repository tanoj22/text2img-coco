"""
Baseline text-to-image generation with Stable Diffusion 1.5.

Picks a handful of captions from data/subset.csv (one per supercategory for
variety), generates one image per caption with a fixed seed, and saves:
  outputs/<image_id>_gen.png      generated images
  outputs/generations.csv         caption, category, seed, settings, timing, paths
  outputs/comparison_grid.png     real COCO image next to the generated one

SD 1.5 uses the same CLIP ViT-L/14 text encoder as Part 2, so each caption is
turned into a 77 x 768 embedding inside the pipeline before denoising.

Usage (from the repo root):
  python src/generate.py
  python src/generate.py --n 8 --steps 30 --seed 42
"""
import argparse
import textwrap
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMG_SIZE = 512


def parse_args():
    p = argparse.ArgumentParser(description="Baseline SD 1.5 generation on the COCO subset")
    p.add_argument("--csv", default=str(ROOT / "data" / "subset.csv"))
    p.add_argument("--out", default=str(ROOT / "outputs"))
    p.add_argument("--n", type=int, default=5, help="number of captions to generate")
    p.add_argument("--steps", type=int, default=50, help="denoising steps (pipeline default is 50)")
    p.add_argument("--guidance", type=float, default=7.5, help="classifier-free guidance scale")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def select_captions(df, n, seed):
    """One caption per supercategory first, then top up at random if n > 12."""
    shuffled = df.sample(frac=1, random_state=seed)
    picks = shuffled.drop_duplicates("supercategory").head(n)
    if len(picks) < n:
        rest = shuffled.drop(picks.index).head(n - len(picks))
        picks = pd.concat([picks, rest])
    return picks.reset_index(drop=True)


def load_pipeline(device, dtype):
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device != "cuda":
        pipe.enable_attention_slicing()  # lowers memory use on MPS and CPU
    pipe.set_progress_bar_config(disable=False)
    return pipe


def get_font(size=16):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def make_grid(rows, out_path, thumb=256, text_h=70):
    """rows: list of (real_path, gen_path, caption). Two columns: real | generated."""
    font = get_font(16)
    header_h = 30
    width = thumb * 2
    height = header_h + len(rows) * (thumb + text_h)
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((10, 6), "COCO original", fill="black", font=font)
    draw.text((thumb + 10, 6), "SD 1.5 generated", fill="black", font=font)

    y = header_h
    for real_path, gen_path, caption in rows:
        for col, path in enumerate([real_path, gen_path]):
            try:
                img = Image.open(path).convert("RGB").resize((thumb, thumb))
            except (FileNotFoundError, OSError):
                img = Image.new("RGB", (thumb, thumb), "lightgray")
            grid.paste(img, (col * thumb, y))
        wrapped = textwrap.fill(caption, width=60)
        draw.multiline_text((8, y + thumb + 4), wrapped, fill="black", font=font, spacing=2)
        y += thumb + text_h
    grid.save(out_path)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    picks = select_captions(df, args.n, args.seed)
    print(f"Selected {len(picks)} captions:")
    for _, r in picks.iterrows():
        print(f"  [{r['category']}] {r['caption']}")

    device, dtype = pick_device()
    print(f"Loading {MODEL_ID} on {device} ({dtype})")
    pipe = load_pipeline(device, dtype)

    records, grid_rows = [], []
    for i, r in picks.iterrows():
        seed = args.seed + i
        # CPU generator keeps results reproducible across devices
        generator = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.time()
        result = pipe(
            r["caption"],
            height=IMG_SIZE,
            width=IMG_SIZE,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )
        elapsed = time.time() - t0
        image = result.images[0]
        flagged = bool(result.nsfw_content_detected and result.nsfw_content_detected[0])

        gen_path = out_dir / f"{r['image_id']}_gen.png"
        image.save(gen_path)
        print(f"  saved {gen_path.name} in {elapsed:.1f}s" + ("  (safety checker flagged)" if flagged else ""))

        real_path = Path(r["processed_path"])
        if not real_path.is_absolute():
            real_path = ROOT / real_path

        records.append({
            "image_id": r["image_id"],
            "category": r["category"],
            "supercategory": r["supercategory"],
            "caption": r["caption"],
            "seed": seed,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "device": device,
            "seconds": round(elapsed, 2),
            "nsfw_flagged": flagged,
            "real_path": str(real_path.relative_to(ROOT)) if real_path.is_relative_to(ROOT) else str(real_path),
            "generated_path": str(gen_path.relative_to(ROOT)) if gen_path.is_relative_to(ROOT) else str(gen_path),
        })
        grid_rows.append((real_path, gen_path, r["caption"]))

    pd.DataFrame(records).to_csv(out_dir / "generations.csv", index=False)
    make_grid(grid_rows, out_dir / "comparison_grid.png")
    print(f"Done. Wrote {len(records)} images, generations.csv and comparison_grid.png to {out_dir}")


if __name__ == "__main__":
    main()
