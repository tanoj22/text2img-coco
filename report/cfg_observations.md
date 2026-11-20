# CFG sweep: observations

Setup: SD 1.5, DDIM, 30 steps, leading spacing, seed 42, 10 prompts from `data/prompts.csv`, guidance scale in {1, 4, 7.5, 12}. CLIP score is torchmetrics CLIPScore with ViT-B/16. Real COCO images score 32.50 against their own captions.

Figure: `report/figures/cfg_grid.png`. Numbers: `report/figures/cfg_summary.csv`.

| CFG | Mean CLIP | Std | Gap to real (32.50) | Mean time (s) |
|----:|----------:|----:|--------------------:|--------------:|
| 1   | 26.40 | 3.23 | -6.10 | 1.87 |
| 4   | 31.26 | 3.46 | -1.24 | 2.46 |
| 7.5 | 31.30 | 3.10 | -1.20 | 2.51 |
| 12  | 32.42 | 3.89 | -0.08 | 2.53 |

## Findings

1. **Guidance is necessary.** At CFG 1 (no guidance) most images are washed-out, sepia-toned scenes with no clear subject. The bird, hydrant, oven, bear and airplane are missing or unrecognizable (p01, p03, p04, p06, p07). Mean CLIP is 26.40, the lowest of the sweep and 6.1 points below real images.

2. **Most of the gain comes from 1 to 4.** Raising CFG from 1 to 4 adds about 4.9 CLIP points. From 4 to 7.5 the mean barely moves (31.26 to 31.30), although the images look different: 7.5 produces cleaner, more saturated subjects that follow attributes more closely. At CFG 4 the "rusted fire hydrant" (p03) is a pipe valve and the "big black bear" (p06) is brown. At 7.5 both are correct.

3. **CFG 12 scores highest but not on every prompt.** Mean CLIP is 32.42, about the same as real images. It beats 7.5 on 7 of 10 prompts, with the largest gains on p04 oven (+3.1), p02 bench (+1.8) and p06 bear (+1.7). Subjects get larger and fill the frame (the p03 hydrant, the p08 dog). Colors and contrast are visibly stronger (the p10 zebra, the p05 kite). That is the usual high-CFG oversaturation. A CLIP score at the real-image level means the image matches the text well, not that it looks photorealistic.

4. **Some prompts do not respond to CFG.** p07 ("an airplane flying in the air during the day") stays at about 25 for every CFG above 1. The plane is rendered over a ground texture rather than open sky. p09 and p10 peak at 7.5 or below. The spread across prompts grows at CFG 12 (std 3.89, the highest of the sweep), so higher guidance helps some captions more than others.

5. **Cost.** CFG 1 is about 25% faster (1.87 s vs about 2.5 s). diffusers skips the unconditional forward pass when the guidance scale is 1 or less, which halves the UNet batch. Between 4 and 12 the cost is flat, so a higher CFG costs no extra time.

## Caveats

- One seed and 10 prompts. The prompt-to-prompt standard deviation is 3 to 4 points, so the standard error of each mean is about 1. The 4 vs 7.5 difference (0.03) is noise. The 7.5 vs 12 difference (1.1) is about one standard error and consistent in direction (7 of 10 prompts), so it is suggestive, not conclusive.
- CLIP score measures text-image alignment only. It does not penalize the oversaturation at CFG 12, so a quality metric such as FID, or human judgment, would be needed to pick the best CFG.
