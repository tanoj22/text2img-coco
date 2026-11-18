"""Pick 10 evaluation prompts from the COCO subset.

Reads data/subset.csv and writes data/prompts.csv. Selection uses seed 42,
keeps one row per COCO category, and prefers captions of 6 to 20 words that
describe one clear main subject.

Usage (from the repo root):
    python src/make_prompts.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUBSET_PATH = ROOT / "data" / "subset.csv"
PROMPTS_PATH = ROOT / "data" / "prompts.csv"
SEED = 42
N_PROMPTS = 10
MIN_WORDS = 6
MAX_WORDS = 20
REQUIRED_COLUMNS = ["image_id", "caption", "category", "processed_path"]

# Tokens that mark more than one main subject.
MULTI_SUBJECT = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|several|many|few|"
    r"multiple|couple|various|pair|pairs|group|groups|crowd|crowds|"
    r"people|both|men|women|children|kids)\b",
    re.IGNORECASE,
)
# Tokens that introduce a pile or a second object.
EXTRA_OBJECT = re.compile(
    r"\b(?:batch|bunch|set|sets|pile|piles|stack|stacks|row|rows|assortment)\b",
    re.IGNORECASE,
)
# COCO labels people with "person", but captions use these words instead.
PERSON_WORDS = {
    "man", "woman", "men", "women", "boy", "girl", "child", "kid",
    "people", "lady", "guy", "skier", "player",
}


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def category_leads(caption: str, category: str) -> bool:
    """True when the category is the opening subject, not a later detail."""
    words = _words(caption)
    parts = _words(category)
    if not parts:
        return False
    for start in range(min(4, len(words))):
        if words[start:start + len(parts)] == parts:
            return True
    return False


def mentions_other_category(caption: str, category: str, categories) -> bool:
    """True when the caption names a COCO category other than its own."""
    text = f" {' '.join(_words(caption))} "
    for other in categories:
        if other == category:
            continue
        base = " ".join(_words(other))
        forms = {base, base + "s", base + "es"}
        if other == "person":
            forms |= PERSON_WORDS
        if any(f" {form} " in text for form in forms):
            return True
    return False


def caption_score(caption: str, category: str, categories=()) -> int:
    """Lower is better. Zero is a 6-20 word caption with one main subject."""
    text = " ".join(str(caption).lower().split())
    n_words = len(text.split())
    score = 0
    if n_words < MIN_WORDS or n_words > MAX_WORDS:
        score += 10
    if "," in text or ";" in text:
        score += 3
    if " and " in text:
        score += 2
    if " with " in text or " next to " in text:
        score += 2
    if MULTI_SUBJECT.search(text):
        score += 4
    if EXTRA_OBJECT.search(text):
        score += 3
    if not category_leads(text, category):
        score += 5
    if mentions_other_category(text, category, categories):
        score += 4
    if "'" in text:
        score += 1
    for word in ("man", "woman", "boy", "girl", "child"):
        if len(re.findall(rf"\b{word}\b", text)) >= 2:
            score += 4
    return score


def load_subset(path: Path = SUBSET_PATH) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"stopping: {path} is missing")
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        sys.exit(
            "stopping: data/subset.csv columns differ. "
            f"missing {missing}. found {list(df.columns)}"
        )
    return df


def select_prompts(df: pd.DataFrame, n: int = N_PROMPTS, seed: int = SEED) -> pd.DataFrame:
    """One row per category. Seed 42 breaks ties among preferred captions."""
    work = df.dropna(subset=REQUIRED_COLUMNS).copy()
    work["caption"] = work["caption"].astype(str).str.strip()
    work = work[work["caption"] != ""]
    categories = sorted(work["category"].astype(str).unique())
    work["score"] = [
        caption_score(caption, category, categories)
        for caption, category in zip(work["caption"], work["category"])
    ]

    # Shuffle first. A stable sort then keeps that order inside each score.
    shuffled = work.sample(frac=1, random_state=seed)
    ordered = shuffled.sort_values("score", kind="mergesort")
    picks = ordered.drop_duplicates("category").head(n).reset_index(drop=True)
    if len(picks) < n:
        sys.exit(f"stopping: needed {n} categories, found {len(picks)}")

    return pd.DataFrame({
        "prompt_id": [f"p{i:02d}" for i in range(1, len(picks) + 1)],
        "image_id": picks["image_id"].astype(int),
        "prompt": picks["caption"].tolist(),
        "category": picks["category"].astype(str).tolist(),
        "real_image_path": (
            picks["processed_path"].astype(str).str.replace("\\", "/", regex=False).tolist()
        ),
    })


def main():
    prompts = select_prompts(load_subset())
    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prompts.to_csv(PROMPTS_PATH, index=False)
    print(f"wrote {len(prompts)} prompts to {PROMPTS_PATH}")
    for _, row in prompts.iterrows():
        print(f"  {row['prompt_id']} [{row['category']}] {row['prompt']}")


if __name__ == "__main__":
    main()
