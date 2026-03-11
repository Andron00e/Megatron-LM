#!/usr/bin/env python3
"""Download and plot 'lm loss' from W&B runs, caching data locally.

Runs sharing the same base name (stripping the final '-<id>' suffix) are
concatenated in order and drawn as a single curve.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import wandb


# ── Configuration ────────────────────────────────────────────────────────────

WANDB_ENTITY = "alehc"
WANDB_PROJECT = "opt_v1"

CACHE_DIR = Path(__file__).resolve().parent / ".wandb_cache"

EXPERIMENTS_TO_PLOT = [
    ## baselines
    "110M-n1",
    # with cosine
    "110M-lr0.004-cos-n1",
    "110M-lr0.003-cos-n1",
    ######## muon
    "110M-muon_m0.95_urm-n1",
    "110M-muon_m0.95-lr0.004-n1",
    ####### ngpt baselines
    "110M-master_a0-wd0-HSrow1_l2_emb_sh-std0.044-L2Norm-fz-nPre-nFin-pst-ppst-usmr-ss-lsS-qklsS-mlplsG-lgslsS-nw-n1",
    # with projection
    "110M-master_a0-wd0-HSrow1_l2_emb_sh_p-std0.044-ngpt-nw-n1",
    # with cosine
    "110M-master_a0-wd0-HSrow1_l2_emb_sh-std0.044-ngpt-nw-cos-n1",
    "110M-master_a0-wd0-HSrow1_l2_emb_sh-lr0.004-std0.044-ngpt-nw-cos-n1",
    ######## hypermuon
    "110M-master_o_b0.9_none-wd0-HSrow1_l2_sh-lr0.004-std0.044-ngpt-nw-n1",
    "110M-master_o_b0.9-wd0-HSrow1_l2_u_sh-lr0.004-std0.044-ngpt-nw-n1",
]

METRIC_KEY = "lm loss"
STEP_KEY = "consumed-tokens"

# ── Helpers ──────────────────────────────────────────────────────────────────


def strip_run_suffix(name: str) -> str:
    """Remove a trailing job/run-id suffix like '-1570020' or '-j1552378'."""
    return re.sub(r"-j?\d{5,}$", "", name)


def cache_path_for(run_id: str) -> Path:
    return CACHE_DIR / f"{run_id}.json"


def fetch_run_history(run, keys: list[str]) -> list[dict]:
    """Download the requested keys from a run, using a local JSON cache."""
    cached = cache_path_for(run.id)
    if cached.exists():
        with open(cached) as f:
            return json.load(f)

    rows = []
    for row in run.scan_history(keys=keys, page_size=5000):
        rows.append({k: row.get(k) for k in keys})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cached, "w") as f:
        json.dump(rows, f)
    return rows


def smooth_and_subsample(
    steps: np.ndarray, vals: np.ndarray, window: int = 50, max_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a moving-average smooth, then uniformly subsample to at most max_points."""
    if len(vals) < 2:
        return steps, vals

    smoothed = uniform_filter1d(vals.astype(float), size=min(window, len(vals)), mode="nearest")

    if len(smoothed) > max_points:
        idx = np.linspace(0, len(smoothed) - 1, max_points, dtype=int)
        return steps[idx], smoothed[idx]
    return steps, smoothed


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=WANDB_ENTITY, help="W&B entity (user or team)"
    )
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Base experiment names to plot (overrides EXPERIMENTS_TO_PLOT)",
    )
    parser.add_argument(
        "--metric", default=METRIC_KEY, help="Metric key to download (default: 'lm loss')"
    )
    parser.add_argument(
        "--step-key", default=STEP_KEY, help="Step key (default: 'iteration')"
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Delete cached data and re-download"
    )
    parser.add_argument(
        "--output", default=None, help="Save figure to this path instead of showing"
    )
    parser.add_argument(
        "--smooth", type=int, default=50,
        help="Moving-average window size (set to 1 to disable smoothing)",
    )
    parser.add_argument(
        "--max-points", type=int, default=2000,
        help="Max data points per curve after subsampling (default: 2000)",
    )
    parser.add_argument(
        "--tail", type=int, default=100,
        help="Number of final raw iterations to average for the ranking table (default: 100)",
    )
    args = parser.parse_args()

    experiments = args.experiments if args.experiments else EXPERIMENTS_TO_PLOT
    if not experiments:
        parser.error(
            "No experiments specified. Pass --experiments or edit EXPERIMENTS_TO_PLOT."
        )

    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"

    # Build a server-side regex so W&B only returns relevant runs.
    name_regex = "|".join(re.escape(name) for name in experiments)
    filters = {"display_name": {"$regex": f"^({name_regex})(-j?\\d{{5,}})?$"}}
    runs = api.runs(path, filters=filters, per_page=1000)

    # Group runs by their base experiment name.
    experiment_runs: dict[str, list] = {name: [] for name in experiments}
    for run in runs:
        base = strip_run_suffix(run.name)
        if base in experiment_runs:
            experiment_runs[base].append(run)
        else:
            print(f"  [warn] Run '{run.name}' matched filter but base '{base}' is unknown")

    # Sort each group so segments are concatenated in order.
    for name in experiment_runs:
        experiment_runs[name].sort(key=lambda r: r.name)

    keys = [args.step_key, args.metric]

    # ── Load & plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ranking: list[tuple[str, float, int]] = []  # (name, tail_mean, n_points)

    for exp_name, run_group in experiment_runs.items():
        if not run_group:
            print(f"  [skip] No runs found for '{exp_name}'")
            continue

        all_steps, all_vals = [], []
        for run in run_group:
            print(f"  Loading {run.name} ({run.id}) …")
            rows = fetch_run_history(run, keys)
            for row in rows:
                s, v = row.get(args.step_key), row.get(args.metric)
                if s is not None and v is not None:
                    all_steps.append(s)
                    all_vals.append(v)

        if not all_steps:
            print(f"  [skip] No data for '{exp_name}'")
            continue

        order = np.argsort(all_steps)
        all_steps = np.array(all_steps)[order]
        all_vals = np.array(all_vals)[order]

        # Deduplicate: for repeated steps keep the value from the latest run.
        # Runs are sorted by name so later restarts come last — keep last occurrence.
        _, unique_idx = np.unique(all_steps[::-1], return_index=True)
        unique_idx = len(all_steps) - 1 - unique_idx  # map back to original order
        unique_idx.sort()
        all_steps = all_steps[unique_idx]
        all_vals = all_vals[unique_idx]

        tail_k = min(args.tail, len(all_vals))
        tail_mean = float(np.mean(all_vals[-tail_k:]))
        ranking.append((exp_name, tail_mean, len(all_vals)))

        plot_steps, plot_vals = smooth_and_subsample(
            all_steps, all_vals, window=args.smooth, max_points=args.max_points
        )
        ax.plot(plot_steps, plot_vals, linewidth=1.2, label=exp_name)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("LM Loss", fontsize=12)
    ax.set_ylim(2.8, 3.5)
    ax.set_title("Language-Model Loss", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    # ── Ranking table ─────────────────────────────────────────────────
    if ranking:
        ranking.sort(key=lambda x: x[1])
        max_name = max(len(r[0]) for r in ranking)
        print(f"\n{'─' * (max_name + 30)}")
        print(f"  Ranking by mean loss over last {args.tail} raw iterations")
        print(f"{'─' * (max_name + 30)}")
        for i, (name, loss, n) in enumerate(ranking, 1):
            print(f"  {i:>2}.  {name:<{max_name}}   {loss:.4f}   ({n:,} pts)")
        print(f"{'─' * (max_name + 30)}")

    if args.output:
        fig.savefig(args.output, dpi=200)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
