"""Data pipeline for the COCO subset.

Owner: Member A (data)

Job: download MS COCO 2017 val, do category-stratified sampling (seed 42),
clean captions, resize images to 512x512, and write data/subset.csv.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ANN_DIR = DATA / "annotations"
IMG_DIR = DATA / "val2017"
OUT_DIR = DATA / "processed"
CSV_PATH = DATA / "subset.csv"
SEED = 42
SIZE = 512


def load_annotations():
    with open(ANN_DIR / "instances_val2017.json") as f:
        inst = json.load(f)
    with open(ANN_DIR / "captions_val2017.json") as f:
        caps = json.load(f)

    file_names = {im["id"]: im["file_name"] for im in inst["images"]}
    categories = {c["id"]: (c["name"], c["supercategory"]) for c in inst["categories"]}

    images_per_cat = defaultdict(set)
    for ann in inst["annotations"]:
        images_per_cat[ann["category_id"]].add(ann["image_id"])

    captions = defaultdict(list)
    for c in caps["annotations"]:
        captions[c["image_id"]].append(c["caption"])

    return file_names, categories, images_per_cat, captions


def stratified_sample(images_per_cat, n_total, per_class, rng):
    label = {}
    rarest_first = sorted(images_per_cat, key=lambda c: (len(images_per_cat[c]), c))

    for cat in rarest_first:
        pool = sorted(images_per_cat[cat] - label.keys())
        for img_id in rng.sample(pool, min(per_class, len(pool))):
            label[img_id] = cat

    remaining = n_total - len(label)
    if remaining > 0:
        leftovers = {}
        for cat in reversed(rarest_first):
            for img_id in images_per_cat[cat]:
                if img_id not in label:
                    leftovers[img_id] = cat
        for img_id in rng.sample(sorted(leftovers), remaining):
            label[img_id] = leftovers[img_id]

    return label


def clean_caption(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(" .")
    return text


def pick_caption(raw_captions, used, rng):
    cleaned = [clean_caption(c) for c in raw_captions]
    valid = [c for c in cleaned if len(c.split()) >= 3]
    fresh = [c for c in valid if c not in used]
    options = fresh or valid
    return rng.choice(options) if options else None


def center_crop_resize(src, dst):
    with Image.open(src) as im:
        im = ImageOps.fit(im.convert("RGB"), (SIZE, SIZE), Image.BICUBIC)
        im.save(dst, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(SEED)
    file_names, categories, images_per_cat, captions = load_annotations()
    label = stratified_sample(images_per_cat, args.n, args.per_class, rng)

    image_ids = sorted(label)
    if args.limit:
        image_ids = image_ids[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, used = [], set()
    for img_id in tqdm(image_ids, desc="processing"):
        caption = pick_caption(captions[img_id], used, rng)
        if caption is None:
            continue
        used.add(caption)
        name, supercat = categories[label[img_id]]
        file_name = file_names[img_id]
        center_crop_resize(IMG_DIR / file_name, OUT_DIR / file_name)
        rows.append({
            "image_id": img_id,
            "file_name": file_name,
            "processed_path": f"data/processed/{file_name}",
            "caption": caption,
            "category": name,
            "supercategory": supercat,
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"wrote {len(df)} rows to {CSV_PATH}")
    print(f"categories: {df['category'].nunique()}  duplicate captions: {df['caption'].duplicated().sum()}")


if __name__ == "__main__":
    main()
