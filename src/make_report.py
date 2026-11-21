"""
Milestone 2 report assets: noise-schedule summary, figures, samples and the
combined training log.

Reads outputs/schedule_mps_log.csv (written by src/run_local.py) and
outputs/generation_log.csv (CFG sweep), and writes:
  report/figures/schedule_summary.csv   one row per scheduler/steps/spacing
  report/figures/schedule_plot.png      CLIP score and time vs steps
  report/figures/schedule_grid.png      same prompt across samplers and steps
  report/samples/<prompt_id>.png        early samples at the chosen config
  report/figures/samples_grid.png       the samples with their prompts
  report/training_log.csv               every generation from both sweeps

Usage (from the repo root):
  python src/make_report.py
"""
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CFG_LOG = ROOT / "outputs" / "generation_log.csv"
SCHED_LOG = ROOT / "outputs" / "schedule_mps_log.csv"
FIG_DIR = ROOT / "report" / "figures"
SAMPLE_DIR = ROOT / "report" / "samples"
TRAINING_LOG = ROOT / "report" / "training_log.csv"
REAL_BASELINE = 32.50

# Config used for the early samples (picked from the schedule sweep, see report)
SAMPLE_CONFIG = dict(scheduler="ddim", steps=30, spacing="trailing", cfg=7.5)
# Prompts shown in the schedule grid (the ones that change most across configs)
GRID_PROMPTS = ["p04", "p06", "p07"]

SAMPLERS = ["ddim", "euler", "dpmpp_karras"]
LABELS = {"ddim": "DDIM", "euler": "Euler", "dpmpp_karras": "DPM++ 2M Karras"}
# categorical slots 1-3 of the reference palette (validated all-pairs)
COLORS = {"ddim": "#2a78d6", "euler": "#eb6834", "dpmpp_karras": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_schedule():
    df = pd.read_csv(SCHED_LOG).drop_duplicates("image_path", keep="last")
    if df.clip_score.isna().any():
        raise SystemExit("schedule log has unscored rows, run src/run_local.py first")
    df["sec_per_step"] = df.time_sec / df.steps
    return df


def summarize(df):
    s = (df.groupby(["scheduler", "steps", "spacing"])
         .agg(clip_mean=("clip_score", "mean"), clip_std=("clip_score", "std"),
              time_mean=("time_sec", "mean"), sec_per_step=("sec_per_step", "mean"),
              n=("clip_score", "size"))
         .reset_index())
    s["vs_real"] = s.clip_mean - REAL_BASELINE
    order = {k: i for i, k in enumerate(SAMPLERS)}
    s = s.sort_values(["spacing", "scheduler", "steps"], key=lambda c: c.map(order) if c.name == "scheduler" else c)
    return s.round(3)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_schedule(summary):
    """Mean CLIP vs steps per sampler (leading spacing). Error bars are the standard error over the
    10 prompts. Time per image is in the summary table, so it is not plotted."""
    lead = summary[summary.spacing == "leading"]
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    offsets = {"ddim": -0.9, "euler": 0.0, "dpmpp_karras": 0.9}  # small x-shift so error bars do not overlap
    style(ax)
    for name in SAMPLERS:
        d = lead[lead.scheduler == name].sort_values("steps")
        se = d.clip_std / d.n ** 0.5
        ax.errorbar(d.steps + offsets[name], d.clip_mean, yerr=se, color=COLORS[name], linewidth=1.8,
                    elinewidth=0.9, capsize=1.8, marker="o", markersize=4.5, markeredgecolor="white",
                    markeredgewidth=1, label=LABELS[name])
    ax.axhline(REAL_BASELINE, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate(f"real images {REAL_BASELINE:.2f}", (35, REAL_BASELINE), xytext=(0, -8),
                textcoords="offset points", fontsize=6.5, color=INK_2)
    ax.set_ylim(30, 35.2)
    ax.set_xticks([20, 30, 50])
    ax.set_xlim(16, 54)
    ax.set_xlabel("denoising steps", fontsize=7, color=INK_2, labelpad=1)
    ax.set_ylabel("mean CLIP score", fontsize=7, color=INK_2, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper center", ncol=3, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 1.13),
              handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG_DIR / "schedule_plot.png", dpi=250)
    plt.close(fig)


def plot_grid(df):
    cols = [(s, n) for s in SAMPLERS for n in (20, 30, 50)]
    lead = df[df.spacing == "leading"].set_index(["prompt_id", "scheduler", "steps"])
    fig, axes = plt.subplots(len(GRID_PROMPTS), len(cols), figsize=(1.25 * len(cols), 1.45 * len(GRID_PROMPTS)),
                             squeeze=False)
    for r, pid in enumerate(GRID_PROMPTS):
        for c, (sched, steps) in enumerate(cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            row = lead.loc[(pid, sched, steps)]
            with Image.open(ROOT / row.image_path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xlabel(f"CLIP {row.clip_score:.1f}", fontsize=9, color=INK_2, labelpad=1.5)
            if r == 0:
                ax.set_title(f"{LABELS[sched].replace('DPM++ 2M Karras', 'DPM++ 2M')}\n{steps} steps",
                             fontsize=9.5, color=INK)
        prompt = lead.loc[(pid, cols[0][0], cols[0][1])].prompt
        axes[r][0].set_ylabel("\n".join(textwrap.wrap(f"{pid}: {prompt}", 16)), fontsize=9,
                              rotation=0, ha="right", va="center", color=INK, labelpad=4)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
    fig.savefig(FIG_DIR / "schedule_grid.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def export_samples(df):
    c = SAMPLE_CONFIG
    picks = df[(df.scheduler == c["scheduler"]) & (df.steps == c["steps"])
               & (df.spacing == c["spacing"]) & (df.cfg == c["cfg"])].sort_values("prompt_id")
    if picks.empty:
        raise SystemExit(f"no rows for sample config {c}")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for row in picks.itertuples():
        shutil.copy(ROOT / row.image_path, SAMPLE_DIR / f"{row.prompt_id}.png")

    n = len(picks)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.95 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, picks.itertuples()):
        with Image.open(ROOT / row.image_path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title("\n".join(textwrap.wrap(f"{row.prompt_id}: {row.prompt}", 30)), fontsize=6.5, color=INK, pad=3)
        ax.text(0.5, -0.04, f"CLIP {row.clip_score:.1f}", transform=ax.transAxes, ha="center", va="top",
                fontsize=6.5, color=INK_2)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.6)
    fig.savefig(FIG_DIR / "samples_grid.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return picks


def training_log(sched):
    cfg = pd.read_csv(CFG_LOG).drop_duplicates("image_path", keep="last")
    cfg.insert(0, "sweep", "cfg")
    cfg["device"] = "cuda"
    sched = sched.drop(columns="sec_per_step").copy()
    sched.insert(0, "sweep", "schedule")
    sched["device"] = "mps"
    log = pd.concat([cfg, sched], ignore_index=True)
    log["time_sec"] = log.time_sec.round(3)
    log["clip_score"] = log.clip_score.round(3)
    log.to_csv(TRAINING_LOG, index=False)
    return log


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_schedule()
    summary = summarize(df)
    summary.to_csv(FIG_DIR / "schedule_summary.csv", index=False)
    print(summary.to_string(index=False))
    plot_schedule(summary)
    plot_grid(df)
    picks = export_samples(df)
    log = training_log(df)
    print(f"{len(picks)} samples -> {SAMPLE_DIR}")
    print(f"training log: {len(log)} rows -> {TRAINING_LOG}")


if __name__ == "__main__":
    main()
