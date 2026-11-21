"""Text for the Milestone 2 report (used by src/report_pdf.py).

Numbers come from report/figures/cfg_summary.csv, report/figures/schedule_summary.csv
and outputs/schedule_mps_log.csv.
"""

C = "<font face='Courier'>{}</font>".format

TEXT = {
    # (scheduler, steps, spacing) highlighted in the schedule table and used for the samples
    "chosen": ("ddim", 30, "trailing"),

    "setup": (
        "Each caption is padded to 77 tokens and encoded by the CLIP ViT-L/14 text encoder of SD 1.5 "
        f"({C('embed.encode')}). The resulting 77 x 768 embedding conditions the UNet through "
        f"{C('prompt_embeds')}, and the embedding of the empty prompt is the unconditional branch for "
        "classifier-free guidance (CFG). We generate 512 x 512 images for 10 evaluation prompts from our "
        "COCO subset (one per category, one clear subject each) with seed 42 and score each image with "
        "CLIP score, using ViT-B/16, a different CLIP model from the one that conditions generation. "
        "Real COCO images score <b>32.50</b> against their own captions, which is our reference."
    ),

    "table_note": (
        "CLIP mean over 10 prompts; std is across prompts, so the standard error of each mean is about 1. "
        "Highlighted rows are the chosen settings. The two sweeps ran on different hardware with different "
        "starting noise, so absolute scores are compared within a table, not across tables."
    ),

    "observations": [
        "<b>Guidance matters far more than the noise schedule.</b> Raising CFG from 1 to 4 adds 4.9 CLIP "
        "points, while all 11 schedule settings fall within 0.9 points of each other (32.28 to 33.22), "
        "about one standard error, and close to the real-image level. CFG 4 and 7.5 tie on CLIP, but 7.5 "
        "gets attributes right (black bear, rusted hydrant); CFG 12 scores highest but oversaturates. "
        "We keep CFG 7.5.",
        "<b>Timestep spacing is the one schedule setting with a consistent effect.</b> At DDIM 30 steps, "
        "trailing spacing beats leading on all 10 prompts (+0.93, standard error 0.22) and linspace on "
        "8 of 10 (+0.90) at no extra cost. Leading spacing starts denoising at t = 958 instead of 999, so "
        "the first step treats pure noise as if it were partly denoised.",
        "<b>DDIM and Euler behave as the same sampler.</b> On the same timesteps they produce almost "
        "identical images (mean pixel difference 1.2 out of 255) and per-prompt CLIP scores within 0.5, "
        "as expected from two first-order deterministic solvers.",
        "<b>DPM++ 2M Karras converges in fewer steps.</b> Its 30-step images differ from its own 50-step "
        "images by 6.7/255 on average, against 16.4 for DDIM and Euler (11.4 vs 19.0 at 20 steps). "
        "When speed matters it is the safest sampler to run at 20 steps.",
        "<b>Cost is linear in steps</b> at about 0.6 s per step on the M4 GPU (12 s at 20 steps, 17 to 19 s "
        "at 30, 28 to 31 s at 50), whatever the sampler. Going to 50 steps costs 1.7x the time of 30 steps "
        "without raising CLIP.",
        "<b>Failure modes.</b> Color and attribute binding are reliable (yellow bus, black bear, brown dog, "
        "rusted hydrant). Errors are in composition and context: the oven prompt gives two ovens full of "
        "pizza and no garbage, the bear has no visible head, the airplane is a small shape in fog (lowest "
        "CLIP, 29.8), and the young zebra grooming itself becomes a black-and-white close-up of stripes.",
    ],

    "next": (
        "<b>CFG 7.5 with DDIM, 30 steps and trailing spacing</b>: the highest mean CLIP in the schedule "
        "sweep (33.22, above the real-image 32.50) at 17.4 s per image. With one seed and 10 prompts the "
        "differences between samplers are within noise, so next we will add seeds and prompts, and add an "
        "image-quality metric such as FID against our real COCO images, since CLIP score measures "
        "text alignment only and cannot separate these settings."
    ),


    "samples_caption": (
        "One image per evaluation prompt at the chosen configuration: SD 1.5, CFG 7.5, DDIM, 30 steps, "
        "trailing spacing, seed 42, 512 x 512, Apple M4 GPU. The value under each image is its CLIP score "
        f"(real images: 32.50). Files: {C('report/samples/p01.png')} to {C('p10.png')}."
    ),
    "grid_caption": (
        "CFG 7.5, leading spacing, seed 42, for the three prompts that change most across settings. "
        "The DDIM and Euler columns are almost identical. DPM++ 2M Karras already looks finished at "
        "20 steps, while DDIM and Euler keep changing up to 50 steps."
    ),
    "cfg_grid_caption": (
        "DDIM, 30 steps, leading spacing, seed 42, CUDA GPU. Rows are the 10 evaluation prompts and "
        "columns the guidance scale; the value under each image is its CLIP score."
    ),

    "log_intro": (
        "SD 1.5 is used pretrained (no weights are updated in this milestone), so the log records every "
        "generation run: its settings, wall-clock time and CLIP score. The same data is in "
        f"{C('report/training_log.csv')}, and the console log of the schedule sweep is "
        f"{C('outputs/schedule_mps_run.log')}."
    ),
    "env": [
        ("Model", "stable-diffusion-v1-5/stable-diffusion-v1-5, fp16, 512 x 512, seed 42"),
        ("Conditioning", "CLIP ViT-L/14 last hidden state, 77 x 768 per prompt; empty-prompt embedding as "
                         "the negative for CFG"),
        ("CFG sweep", "DDIM, 30 steps, leading spacing, CFG 1 / 4 / 7.5 / 12; CUDA GPU; 40 images"),
        ("Schedule sweep", "CFG 7.5; DDIM, Euler, DPM++ 2M Karras at 20 / 30 / 50 steps (leading), plus "
                           "DDIM 30 steps with trailing and linspace spacing; 110 images in 36.1 min"),
        ("Schedule hardware", "Apple M4 GPU (MPS), fp16, torch 2.14.1, diffusers 0.35.2, "
                              "transformers 4.57.6"),
        ("Metric", "torchmetrics CLIPScore with openai/clip-vit-base-patch16; real-image baseline 32.50"),
    ],
    "log_rows_note": (
        "Prompt ids are the ones on the Early samples page. Time is the denoising loop only (model loading "
        "excluded). CFG-sweep times are on a CUDA GPU and schedule-sweep times on the Apple M4 GPU, so the "
        "two are not directly comparable."
    ),
}
