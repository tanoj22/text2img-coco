"""
Run SD 1.5 generations over data/prompts.csv and log every image.

Single config:
  python src/run_experiment.py --scheduler ddim --steps 30 --cfg 7.5
  python src/run_experiment.py --scheduler euler --steps 20 --cfg 7.5 --spacing trailing

Presets:
  python src/run_experiment.py --sweep cfg
  python src/run_experiment.py --sweep schedule

Resume: an image is skipped when its png exists AND its image_path is already
in outputs/generation_log.csv. If the png exists but was never logged (for
example a crash between save and log), it is regenerated and logged.
Image paths do not encode the seed, so do not mix seeds in one outputs folder.

Paths in logger.py are relative, so this script switches to the repo root
before doing anything. It can be launched from any directory.
"""
import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_CSV = ROOT / "data" / "prompts.csv"

SWEEPS = {
    "cfg": [("ddim", 30, "leading", cfg) for cfg in (1.0, 4.0, 7.5, 12.0)],
    "schedule": [
        (sched, steps, "leading", 7.5)
        for sched in ("ddim", "euler", "dpmpp_karras")
        for steps in (20, 30, 50)
    ]
    + [("ddim", 30, "trailing", 7.5), ("ddim", 30, "linspace", 7.5)],
}


def parse_args():
    p = argparse.ArgumentParser(description="Milestone 2 generation experiments")
    p.add_argument("--scheduler", choices=["ddim", "euler", "dpmpp_karras"])
    p.add_argument("--steps", type=int)
    p.add_argument("--spacing", default="leading", choices=["leading", "trailing", "linspace"])
    p.add_argument("--cfg", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", choices=sorted(SWEEPS))
    args = p.parse_args()
    if args.sweep is None:
        missing = [n for n in ("scheduler", "steps", "cfg") if getattr(args, n) is None]
        if missing:
            p.error("without --sweep these are required: " + ", ".join("--" + m for m in missing))
    return args


def read_prompts():
    with open(PROMPTS_CSV, newline="", encoding="utf-8") as f:
        return [(row["prompt_id"], row["prompt"]) for row in csv.DictReader(f)]


def logged_image_paths(log_path):
    if not Path(log_path).exists():
        return set()
    with open(log_path, newline="", encoding="utf-8") as f:
        return {row["image_path"] for row in csv.DictReader(f)}


def run_config(scheduler, steps, spacing, cfg, seed, prompts, done):
    import logger
    from pipeline import generate  # imported here so a fully resumed run never loads the model

    label = f"{scheduler} s{steps} {spacing} cfg{cfg}"
    for prompt_id, prompt in prompts:
        image_path = logger.image_path_for(scheduler, steps, spacing, cfg, prompt_id)
        if Path(image_path).exists() and image_path in done:
            print(f"[{label}] {prompt_id} skipped")
            continue

        image, time_sec = generate(
            prompt, scheduler=scheduler, steps=steps, spacing=spacing, cfg=cfg, seed=seed
        )
        image.save(image_path)
        logger.log_generation({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "seed": seed,
            "scheduler": scheduler,
            "steps": steps,
            "spacing": spacing,
            "cfg": cfg,
            "time_sec": round(time_sec, 3),
            "clip_score": None,
            "image_path": image_path,
        })
        done.add(image_path)
        print(f"[{label}] {prompt_id} generated in {time_sec:.2f}s")


def main():
    args = parse_args()
    os.chdir(ROOT)

    import logger

    prompts = read_prompts()
    configs = SWEEPS[args.sweep] if args.sweep else [(args.scheduler, args.steps, args.spacing, args.cfg)]
    done = logged_image_paths(logger.LOG_PATH)

    print(f"{len(prompts)} prompts x {len(configs)} configs, seed {args.seed}")
    for scheduler, steps, spacing, cfg in configs:
        run_config(scheduler, steps, spacing, float(cfg), args.seed, prompts, done)
    print(f"done. log: {logger.LOG_PATH}")


if __name__ == "__main__":
    main()
