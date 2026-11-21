# Noise schedule sweep: observations

Setup: SD 1.5, CFG 7.5, seed 42, 10 prompts from `data/prompts.csv`, the `--sweep schedule` preset from `src/run_experiment.py` (DDIM, Euler and DPM++ 2M Karras at 20, 30 and 50 steps with leading spacing, plus DDIM 30 steps with trailing and linspace spacing). Run on an Apple M4 GPU (MPS, fp16) with `src/run_local.py`: 110 images in 36.1 min. CLIP score is the same torchmetrics CLIPScore (ViT-B/16) used for the CFG sweep. Real COCO images score 32.50.

Figures: `report/figures/schedule_plot.png`, `report/figures/schedule_grid.png`. Numbers: `report/figures/schedule_summary.csv`. Per-image log: `outputs/schedule_mps_log.csv`.

| Sampler | Steps | Spacing | Mean CLIP | Std | Mean time (s) |
|---|---:|---|---:|---:|---:|
| DDIM | 20 | leading | 32.92 | 2.85 | 12.6 |
| DDIM | 30 | leading | 32.28 | 3.14 | 19.1 |
| DDIM | 50 | leading | 33.09 | 2.75 | 30.9 |
| Euler | 20 | leading | 32.89 | 2.88 | 12.9 |
| Euler | 30 | leading | 32.39 | 3.27 | 19.1 |
| Euler | 50 | leading | 32.98 | 2.82 | 29.4 |
| DPM++ 2M Karras | 20 | leading | 32.58 | 2.89 | 12.1 |
| DPM++ 2M Karras | 30 | leading | 33.08 | 2.90 | 17.5 |
| DPM++ 2M Karras | 50 | leading | 32.82 | 2.65 | 28.4 |
| DDIM | 30 | linspace | 33.18 | 2.76 | 17.4 |
| **DDIM** | **30** | **trailing** | **33.22** | **2.85** | **17.4** |

## Findings

1. **The schedule barely moves CLIP.** All 11 settings fall between 32.28 and 33.22. With a prompt-to-prompt std near 3, the standard error of each mean is about 0.9, so no sampler or step count is clearly better on text alignment. For comparison, CFG 1 to 4 moved CLIP by 4.9 points.

2. **Trailing spacing is the one consistent gain.** At DDIM 30 steps, trailing beats leading on all 10 prompts (mean +0.93, standard error of the paired difference 0.22). Linspace beats it on 8 of 10 (+0.90). Both cost the same as leading. With 30 steps, leading spacing uses timesteps 958, 925, ..., 1, so it never starts at t = 999 even though the initial latent is pure noise; trailing (999, 966, ..., 32) and linspace (999, 965, ..., 0) do.

3. **DDIM and Euler are interchangeable.** On the same timesteps they give almost the same image (mean absolute pixel difference 1.2 out of 255) and per-prompt CLIP within 0.5. Both are first-order deterministic solvers of the same ODE.

4. **DPM++ 2M Karras converges fastest.** Mean absolute pixel difference from the same sampler's 50-step image:

   | Sampler | 20 steps | 30 steps |
   |---|---:|---:|
   | DDIM | 19.0 | 16.4 |
   | Euler | 19.1 | 16.4 |
   | DPM++ 2M Karras | 11.4 | 6.7 |

   DPM++ images barely change after 20 to 30 steps. DDIM and Euler are still changing at 30 (see p04 and p07 in `schedule_grid.png`).

5. **Cost is linear in steps.** About 0.6 s per step on the M4 regardless of sampler (DPM++ is slightly cheaper per step). 50 steps costs 1.7x the time of 30 with no CLIP gain.

6. **Failure modes at the chosen setting** (`report/samples/`): attributes and colors are right (yellow bus, black bear, brown dog, rusted hydrant), but composition and context slip. p04 shows two ovens with pizza and no garbage, p06 has a bear with no visible head, p07 is a small plane in fog (lowest CLIP, 29.8), and p10 is a black-and-white close-up of zebra stripes rather than a young zebra grooming.

## Chosen setting

CFG 7.5, DDIM, 30 steps, trailing spacing: highest mean CLIP (33.22) at 17.4 s per image. If generation time matters more, DPM++ 2M Karras at 20 steps (12.1 s) is the best cut because it is closest to converged.

## Caveats

- One seed and 10 prompts. Differences between samplers are within one standard error; only the spacing result holds on every prompt.
- This sweep ran on an Apple M4 with a CPU-seeded generator, while the CFG sweep ran on CUDA with a CUDA generator. The starting noise differs, so DDIM 30 leading CFG 7.5 scores 32.28 here and 31.30 in the CFG sweep. Compare numbers within a sweep, not across sweeps.
- CLIP score measures text alignment, not image quality. An image-quality metric such as FID against real COCO images is needed to rank these settings properly.
