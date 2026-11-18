"""Append-only log of text-to-image generation runs."""

import csv
from pathlib import Path

LOG_PATH = "outputs/generation_log.csv"
COLUMNS = [
    "prompt_id",
    "prompt",
    "seed",
    "scheduler",
    "steps",
    "spacing",
    "cfg",
    "time_sec",
    "clip_score",
    "image_path",
]


def log_generation(row: dict) -> None:
    """Append one generation record. Write the header if the file does not exist.

    clip_score may be None when the image has not been scored yet.
    Raises ValueError if the keys are not exactly COLUMNS.
    """
    keys = set(row)
    if keys != set(COLUMNS):
        missing = [col for col in COLUMNS if col not in keys]
        extra = sorted(keys - set(COLUMNS))
        raise ValueError(
            f"row keys must match COLUMNS exactly. missing={missing} extra={extra}"
        )

    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    record = {col: row[col] for col in COLUMNS}
    if record["clip_score"] is None:
        record["clip_score"] = ""

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)


def image_path_for(scheduler, steps, spacing, cfg, prompt_id) -> str:
    """Return the png path for one run and create its folder.

    outputs/{scheduler}_s{steps}_{spacing}_cfg{cfg}/{prompt_id}.png
    """
    folder = f"outputs/{scheduler}_s{steps}_{spacing}_cfg{cfg}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    return f"{folder}/{prompt_id}.png"
