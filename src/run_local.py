"""
Run a run_experiment.py sweep preset on Apple Silicon (MPS) or CPU.

pipeline.py is CUDA only (CUDA generator, cuda.synchronize, fp16). This script
runs the same sweep presets with the same schedulers (pipeline.SCHEDULERS) and
the same CLIP ViT-L/14 conditioning (embed.encode -> prompt_embeds), but on
whatever device is available. Results go to a separate log so runs from
different devices are never mixed with outputs/generation_log.csv.

Only the fp16 weights of the text encoder, UNet and VAE are loaded. The safety
checker is left out, which saves about 1.2 GB of download and avoids blacked-out
false positives on harmless COCO captions.

Outputs (for --sweep schedule):
  outputs/schedule_mps/<scheduler>_s<steps>_<spacing>_cfg<cfg>/<prompt_id>.png
  outputs/schedule_mps_log.csv   same columns as outputs/generation_log.csv
  outputs/schedule_mps_run.log   run log with device, versions and timings

Usage (from any directory):
  python src/run_local.py --sweep schedule          # generate, then CLIP-score
  python src/run_local.py --sweep schedule --limit 1  # quick check, 1 prompt
"""
import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path

import diffusers
import torch
import transformers
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logger  # noqa: E402
from embed import EMBED_DIM, MAX_LEN, encode  # noqa: E402
from generate import IMG_SIZE, MODEL_ID  # noqa: E402
from pipeline import SCHEDULERS  # noqa: E402
from run_experiment import SWEEPS, logged_image_paths, read_prompts  # noqa: E402

log = logging.getLogger("run_local")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def load_pipe(device, dtype):
    variant = "fp16" if dtype == torch.float16 else None
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant=variant,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def setup_logging(path):
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for handler in (logging.FileHandler(path, mode="a", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def main():
    p = argparse.ArgumentParser(description="Run a sweep preset on MPS or CPU")
    p.add_argument("--sweep", choices=sorted(SWEEPS), default="schedule")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    p.add_argument("--no-score", action="store_true", help="skip CLIP scoring at the end")
    args = p.parse_args()

    os.chdir(ROOT)  # logger and evaluate use paths relative to the repo root
    device, dtype = pick_device()
    tag = f"{args.sweep}_{device}"
    out_dir = Path("outputs") / tag
    log_csv = f"outputs/{tag}_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(f"outputs/{tag}_run.log")
    logger.LOG_PATH = log_csv  # log_generation reads this at call time

    prompts = read_prompts()[: args.limit]
    configs = SWEEPS[args.sweep]
    done = logged_image_paths(log_csv)

    log.info("=" * 70)
    log.info(f"sweep={args.sweep} prompts={len(prompts)} configs={len(configs)} seed={args.seed}")
    log.info(f"device={device} dtype={str(dtype).replace('torch.', '')} machine={platform.machine()} "
             f"torch={torch.__version__} diffusers={diffusers.__version__} "
             f"transformers={transformers.__version__}")

    t_load = time.perf_counter()
    pipe = load_pipe(device, dtype)
    base_config = dict(pipe.scheduler.config)
    log.info(f"loaded {MODEL_ID} in {time.perf_counter() - t_load:.1f}s")

    def embed(texts):
        hidden, _ = encode(texts, pipe.tokenizer, pipe.text_encoder)
        assert hidden.shape[1:] == (MAX_LEN, EMBED_DIM), hidden.shape
        return hidden.to(device, dtype)

    neg = embed([""])
    total = 0.0
    for scheduler, steps, spacing, cfg in configs:
        cfg = float(cfg)
        cls, extra = SCHEDULERS[scheduler]
        pipe.scheduler = cls.from_config(base_config, timestep_spacing=spacing, **extra)
        folder = out_dir / f"{scheduler}_s{steps}_{spacing}_cfg{cfg}"
        folder.mkdir(parents=True, exist_ok=True)
        label = f"{scheduler:>12} s{steps:<2} {spacing:<8} cfg{cfg:g}"
        config_times = []

        for prompt_id, prompt in prompts:
            image_path = (folder / f"{prompt_id}.png").as_posix()
            if Path(image_path).exists() and image_path in done:
                log.info(f"[{label}] {prompt_id} skipped (already logged)")
                continue

            generator = torch.Generator("cpu").manual_seed(args.seed)
            sync(device)
            t0 = time.perf_counter()
            image = pipe(
                prompt_embeds=embed([prompt]),
                negative_prompt_embeds=neg,
                height=IMG_SIZE,
                width=IMG_SIZE,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            ).images[0]
            sync(device)
            secs = time.perf_counter() - t0

            image.save(image_path)
            logger.log_generation({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": args.seed,
                "scheduler": scheduler,
                "steps": steps,
                "spacing": spacing,
                "cfg": cfg,
                "time_sec": round(secs, 3),
                "clip_score": None,
                "image_path": image_path,
            })
            done.add(image_path)
            config_times.append(secs)
            total += secs
            log.info(f"[{label}] {prompt_id} {secs:6.2f}s  {secs / steps:.3f}s/step")

        if config_times:
            mean = sum(config_times) / len(config_times)
            log.info(f"[{label}] config done: {len(config_times)} images, mean {mean:.2f}s/image")

    log.info(f"generation done in {total / 60:.1f} min. log: {log_csv}")

    if not args.no_score:
        from evaluate import score_log

        del pipe
        if device == "mps":
            torch.mps.empty_cache()
        t0 = time.perf_counter()
        df = score_log(log_csv)
        log.info(f"CLIP-scored {len(df)} rows in {time.perf_counter() - t0:.1f}s "
                 f"(mean {df.clip_score.mean():.2f})")


if __name__ == "__main__":
    main()
