"""
Milestone 2 generation pipeline for Stable Diffusion 1.5.

Reuses load_pipeline() from src/generate.py for model loading and encode()
from src/embed.py for the CLIP ViT-L/14 text conditioning. The model is loaded
once on first use and reused for every call to generate().

Prompts are encoded with embed.encode() using the pipeline's own tokenizer and
text encoder, so the (N, 77, 768) last hidden state passed as prompt_embeds is
exactly what SD 1.5 would compute internally, without loading a second copy of
the text encoder.

Usage (from the repo root):
  python src/pipeline.py      # writes outputs/test_pipeline.png
"""
import copy
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler

from embed import EMBED_DIM, MAX_LEN, encode
from generate import IMG_SIZE, load_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda"
DTYPE = torch.float16

VALID_SPACINGS = ("leading", "trailing", "linspace")

# name -> (scheduler class, extra kwargs passed to from_config)
# Note: with use_karras_sigmas=True, DPMSolverMultistepScheduler derives its
# timesteps from the Karras sigma schedule, so timestep_spacing has no effect
# on "dpmpp_karras". It is still passed so every scheduler is built the same way.
SCHEDULERS = {
    "ddim": (DDIMScheduler, {}),
    "euler": (EulerDiscreteScheduler, {}),
    "dpmpp_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
}

_PIPE = None
_ORIGINAL_SCHEDULER_CONFIG = None
_NEGATIVE_EMBEDS = None


def get_pipe():
    """Load SD 1.5 once (fp16 on CUDA, attention slicing on) and cache it."""
    global _PIPE, _ORIGINAL_SCHEDULER_CONFIG
    if _PIPE is None:
        if not torch.cuda.is_available():
            raise RuntimeError("pipeline.py requires a CUDA GPU")
        pipe = load_pipeline(DEVICE, DTYPE)
        # generate.py only enables slicing off CUDA, so turn it on here
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        # Frozen copy so every scheduler is built from the untouched original
        _ORIGINAL_SCHEDULER_CONFIG = copy.deepcopy(pipe.scheduler.config)
        _PIPE = pipe
    return _PIPE


def make_scheduler(name, spacing):
    """Build a fresh scheduler from the original config. Never stacks changes."""
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {name!r}, expected one of {list(SCHEDULERS)}")
    if spacing not in VALID_SPACINGS:
        raise ValueError(f"unknown spacing {spacing!r}, expected one of {list(VALID_SPACINGS)}")
    get_pipe()
    cls, extra = SCHEDULERS[name]
    return cls.from_config(_ORIGINAL_SCHEDULER_CONFIG, timestep_spacing=spacing, **extra)


def embed_prompts(prompts):
    """Return (N, 77, 768) fp16 CUDA embeddings via embed.encode()."""
    pipe = get_pipe()
    hidden, _pooled = encode(prompts, pipe.tokenizer, pipe.text_encoder)
    if hidden.shape[1:] != (MAX_LEN, EMBED_DIM):
        raise RuntimeError(f"unexpected embedding shape {tuple(hidden.shape)}")
    return hidden.to(DEVICE, DTYPE)


def negative_embeds(batch_size):
    """Embedding of the empty prompt, cached and expanded to the batch size."""
    global _NEGATIVE_EMBEDS
    if _NEGATIVE_EMBEDS is None:
        _NEGATIVE_EMBEDS = embed_prompts([""])
    return _NEGATIVE_EMBEDS.expand(batch_size, -1, -1)


def generate(prompt, scheduler="ddim", steps=30, spacing="leading", cfg=7.5, seed=42):
    """Generate one 512x512 image. Returns (PIL.Image, time_sec).

    time_sec covers only the denoising call (model loading excluded), with
    CUDA synchronized on both ends so the timing is accurate.

    Note: diffusers disables classifier-free guidance when cfg <= 1.0, so the
    negative embeddings are unused at cfg=1.0.
    """
    pipe = get_pipe()
    pipe.scheduler = make_scheduler(scheduler, spacing)

    prompt_embeds = embed_prompts([prompt])
    neg_embeds = negative_embeds(prompt_embeds.shape[0])
    generator = torch.Generator(DEVICE).manual_seed(seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        height=IMG_SIZE,
        width=IMG_SIZE,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    torch.cuda.synchronize()
    time_sec = time.perf_counter() - t0

    return result.images[0], time_sec


if __name__ == "__main__":
    out_path = ROOT / "outputs" / "test_pipeline.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image, secs = generate("a giraffe standing next to a tree")
    image.save(out_path)
    print(f"saved {out_path} in {secs:.2f}s")
