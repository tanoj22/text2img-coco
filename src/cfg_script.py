"""
CFG sweep figure and summary from outputs/generation_log.csv.

Rows are prompts, columns are guidance scales (DDIM, 30 steps, leading spacing,
the --sweep cfg preset). Each cell shows its CLIP score; each column header
shows the mean. Run after the sweep and evaluate.score_log().

Usage (from the repo root):
  python src/cfg_script.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "generation_log.csv"
OUT_FIG = ROOT / "report" / "figures" / "cfg_grid.png"
OUT_CSV = ROOT / "report" / "figures" / "cfg_summary.csv"
REAL_BASELINE = 32.50


def main():
    df = pd.read_csv(LOG)
    df = df[(df.scheduler == "ddim") & (df.steps == 30) & (df.spacing == "leading")]
    # keep the latest row if a config was ever logged twice
    df = df.drop_duplicates("image_path", keep="last")
    if df.empty or df.clip_score.isna().any():
        raise SystemExit("run the cfg sweep and evaluate.score_log() first")

    summary = df.groupby("cfg").agg(
        clip_mean=("clip_score", "mean"),
        clip_std=("clip_score", "std"),
        time_mean=("time_sec", "mean"),
        n=("clip_score", "size"),
    ).round(3)
    summary["vs_real"] = (summary.clip_mean - REAL_BASELINE).round(3)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV)
    print(summary.to_string())
    print(f"real-image baseline: {REAL_BASELINE}")

    cfgs = sorted(df.cfg.unique())
    prompts = df.drop_duplicates("prompt_id").sort_values("prompt_id")[["prompt_id", "prompt"]]
    cells = df.set_index(["prompt_id", "cfg"])

    fig, axes = plt.subplots(len(prompts), len(cfgs), figsize=(2.2 * len(cfgs) + 2.6, 2.2 * len(prompts)),
                             squeeze=False)
    for r, (pid, prompt) in enumerate(prompts.itertuples(index=False)):
        for c, cfg in enumerate(cfgs):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if (pid, cfg) not in cells.index:
                ax.set_facecolor("lightgray")
                continue
            row = cells.loc[(pid, cfg)]
            with Image.open(ROOT / row.image_path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xlabel(f"CLIP {row.clip_score:.1f}", fontsize=8, labelpad=2)
            if r == 0:
                ax.set_title(f"CFG {cfg:g}\nmean CLIP {summary.loc[cfg, 'clip_mean']:.2f}", fontsize=10)
        axes[r][0].set_ylabel(f"{pid}: {prompt[:38]}" + ("..." if len(prompt) > 38 else ""),
                              fontsize=8, rotation=0, ha="right", va="center", wrap=True)

    fig.suptitle(f"SD 1.5, DDIM 30 steps, leading spacing, seed 42 (real-image CLIP baseline {REAL_BASELINE})",
                 fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))  # leave room so the suptitle clears the column titles
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
