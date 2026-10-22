
"""Inference-only PILNODE evaluation against a configurable analytical simulator.

Features:
- Any checkpoint source count.
- Arbitrary physical evaluation times within the trained horizon.
- Optional ground-truth source overrides.
- Optimal source matching with explicit unmatched-source reporting.
- Adaptive heatmap normalization and robust filenames.
- CSV reports for source recovery and temperature errors.

Examples:
    python modelviz.py
    python modelviz.py --times 0.01 0.07 1 17.5 100 500
    python modelviz.py --grid 301 --device cpu
    python modelviz.py --truth-sources -0.3 -0.25 7000 0.3 0.25 5000
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
import config as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import PowerNorm
from scipy.optimize import linear_sum_assignment
from scipy.special import exp1

from model import Config, PILNODE
from simulator import (
    Ti,
    alpha,
    k,
    source_radius,
    sources as DEFAULT_SOURCES,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--truth-sources",
        nargs="+",
        type=float,
        default=None,
        metavar="VALUE",
        help="Optional flat sequence of x y q triples; overrides simulator sources.",
    )
    p.add_argument("--checkpoint", default=C.CHECKPOINT)
    p.add_argument("--out-dir", default=C.VIZ_DIR)
    p.add_argument("--times", nargs="+", type=float, default=C.VIZ_TIMES)
    p.add_argument("--grid", type=int, default=C.VIZ_GRID)
    p.add_argument("--xmin", type=float, default=C.X_MIN)
    p.add_argument("--xmax", type=float, default=C.X_MAX)
    p.add_argument("--ymin", type=float, default=C.Y_MIN)
    p.add_argument("--ymax", type=float, default=C.Y_MAX)
    p.add_argument("--error-vmax", type=float, default=C.VIZ_ERROR_VMAX)
    p.add_argument("--dpi", type=int, default=C.VIZ_DPI)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def normalize_sources(value, name):
    """Return an (N, 3) float64 array of x, y, q triples."""
    arr = np.asarray(value, dtype=np.float64)

    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    if arr.ndim == 2 and arr.shape[1] == 3:
        result = arr.reshape(-1, 3)
    elif arr.size % 3 == 0:
        result = arr.reshape(-1, 3)
    else:
        raise ValueError(
            f"{name} must contain complete (x, y, q) triples; "
            f"received shape {arr.shape}."
        )

    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")

    return result


def analytical_temperature(X, Y, t, source_list):
    """Evaluate the analytical field for any finite collection of sources."""
    result = np.full(np.broadcast_shapes(X.shape, Y.shape), Ti, dtype=np.float64)

    if t <= 0:
        return result

    for xs, ys, q in source_list:
        r2 = (X - xs) ** 2 + (Y - ys) ** 2 + source_radius**2
        z = r2 / (4.0 * alpha * t)
        result += (q / (4.0 * np.pi * k)) * exp1(z)

    return result


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "Checkpoint must contain model_state_dict and config. "
            "Retrain with the current model.py."
        )

    cfg = Config(**checkpoint["config"])
    model = PILNODE(cfg).to(device)

    state = checkpoint["model_state_dict"]
    model_state = model.state_dict()

    # Support checkpoints whose source parameters were stored as scalars.
    state = dict(state)
    for name in ("xs", "ys", "log_q"):
        if name in state and name in model_state:
            state[name] = state[name].reshape_as(model_state[name])

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, cfg, checkpoint


def choose_times(requested, t_max):
    """Return sorted, unique, valid physical evaluation times."""
    if requested is None:
        # Geometric coverage works for short and long time horizons.
        fractions = np.geomspace(0.01, 1.0, 7)
        times = fractions * t_max
    else:
        times = np.asarray(requested, dtype=np.float64)

    if times.size == 0:
        raise ValueError("At least one evaluation time is required.")

    if not np.isfinite(times).all():
        raise ValueError("Evaluation times must be finite.")

    if np.any(times <= 0):
        raise ValueError("All evaluation times must be positive.")

    # Remove duplicates and sort for predictable output ordering.
    times = np.unique(times)

    valid = times[times <= t_max]
    invalid = times[times > t_max]

    for t in invalid:
        print(
            f"Skipping t={t:g}: exceeds checkpoint horizon "
            f"t_max={t_max:g}."
        )

    if valid.size == 0:
        raise ValueError(
            f"No requested times fall within (0, {t_max:g}]."
        )

    return valid


def source_recovery(estimated, truth):
    """Match sources by spatial distance; source order is not identifiable."""
    if len(estimated) == 0 or len(truth) == 0:
        return [], list(range(len(estimated))), list(range(len(truth)))

    cost = np.linalg.norm(
        estimated[:, None, :2] - truth[None, :, :2],
        axis=-1,
    )

    rows, cols = linear_sum_assignment(cost)

    pairs = []
    for i, j in zip(rows, cols):
        ex, ey, eq = estimated[i]
        tx, ty, tq = truth[j]

        pairs.append({
            "estimated_index": int(i),
            "truth_index": int(j),
            "estimated_x": ex,
            "truth_x": tx,
            "x_error": ex - tx,
            "estimated_y": ey,
            "truth_y": ty,
            "y_error": ey - ty,
            "estimated_q": eq,
            "truth_q": tq,
            "q_error": eq - tq,
            "q_relative_error": (
                (eq - tq) / abs(tq) if tq != 0 else np.nan
            ),
            "position_error": float(np.hypot(ex - tx, ey - ty)),
        })

    unmatched_est = sorted(set(range(len(estimated))) - set(rows.tolist()))
    unmatched_truth = sorted(set(range(len(truth))) - set(cols.tolist()))

    return pairs, unmatched_est, unmatched_truth


def safe_time_name(t):
    """Create filesystem-safe names without losing time distinctions."""
    return f"{t:.12g}".replace("-", "m").replace("+", "").replace(".", "p")


def main():
    args = parse_args()

    if args.grid < 2:
        raise ValueError("--grid must be at least 2.")

    if not (
        np.isfinite([args.xmin, args.xmax, args.ymin, args.ymax]).all()
        and args.xmin < args.xmax
        and args.ymin < args.ymax
    ):
        raise ValueError("Grid bounds must be finite and strictly increasing.")

    if args.error_vmax is not None and args.error_vmax <= 0:
        raise ValueError("--error-vmax must be positive.")

    if args.truth_sources is not None:
        truth = normalize_sources(args.truth_sources, "--truth-sources")
    else:
        truth = normalize_sources(DEFAULT_SOURCES, "simulator.sources")

    device = torch.device(args.device)
    model, cfg, checkpoint = load_model(Path(args.checkpoint), device)

    estimated_dict = model.source_estimate()
    estimated = np.column_stack([
        np.asarray(estimated_dict[name], dtype=np.float64).reshape(-1)
        for name in ("x", "y", "q")
    ])

    if not np.isfinite(estimated).all():
        raise ValueError("The checkpoint produced non-finite source estimates.")

    if len(estimated) != cfg.n_sources:
        raise ValueError(
            f"Checkpoint declares {cfg.n_sources} sources, "
            f"but model returned {len(estimated)} estimates."
        )

    times = choose_times(args.times, cfg.t_max)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Model source count: {len(estimated)}")
    print(f"Truth source count: {len(truth)}")
    print(f"Training time horizon: (0, {cfg.t_max:g}]")
    print(f"Evaluation times: {times.tolist()}")

    pairs, unmatched_est, unmatched_truth = source_recovery(estimated, truth)

    print("\nSource recovery:")
    for row in pairs:
        print(
            f"  estimate[{row['estimated_index']}] -> "
            f"truth[{row['truth_index']}]: "
            f"position error={row['position_error']:.4e}, "
            f"q relative error={row['q_relative_error']:+.3%}"
        )

    if unmatched_est:
        print(f"  Unmatched estimated sources: {unmatched_est}")
    if unmatched_truth:
        print(f"  Unmatched analytical sources: {unmatched_truth}")

    if len(estimated) != len(truth):
        print(
            "WARNING: Source counts differ. "
            "Matched-source metrics do not evaluate missing or extra sources."
        )

    source_csv = out_dir / "source_recovery.csv"
    pd.DataFrame(pairs).to_csv(source_csv, index=False)

    x = np.linspace(args.xmin, args.xmax, args.grid)
    y = np.linspace(args.ymin, args.ymax, args.grid)
    X, Y = np.meshgrid(x, y)
    extent = [args.xmin, args.xmax, args.ymin, args.ymax]

    xt = torch.as_tensor(
        X.reshape(-1, 1), dtype=torch.float32, device=device
    )
    yt = torch.as_tensor(
        Y.reshape(-1, 1), dtype=torch.float32, device=device
    )

    metrics = []

    for t in times:
        tt = torch.full_like(xt, float(t))

        with torch.inference_mode():
            pred = (
                model(xt, yt, tt)
                .detach()
                .cpu()
                .numpy()
                .reshape(X.shape)
            )

        ref = analytical_temperature(X, Y, float(t), truth)

        if not np.isfinite(pred).all():
            print(f"Skipping t={t:g}: model prediction contains non-finite values.")
            continue

        difference = pred - ref
        absolute_error = np.abs(difference)

        rmse = float(np.sqrt(np.mean(difference**2)))
        mae = float(np.mean(absolute_error))
        max_error = float(np.max(absolute_error))
        reference_range = float(np.ptp(ref))

        metrics.append({
            "time": float(t),
            "rmse_degC": rmse,
            "mae_degC": mae,
            "max_abs_error_degC": max_error,
            "reference_min_degC": float(np.min(ref)),
            "reference_max_degC": float(np.max(ref)),
            "reference_range_degC": reference_range,
            "n_grid_points": int(ref.size),
            "n_truth_sources": int(len(truth)),
            "n_estimated_sources": int(len(estimated)),
        })

        # Shared normalization makes reference/prediction comparable per frame.
        rise_max = max(
            float(np.max(ref - Ti)),
            float(np.max(pred - Ti)),
            1e-8,
        )
        heat_norm = PowerNorm(
            gamma=0.4,
            vmin=0.0,
            vmax=rise_max,
        )

        error_max = (
            args.error_vmax
            if args.error_vmax is not None
            else max(max_error, 1e-8)
        )

        fig, axes = plt.subplots(
            1, 3, figsize=(16, 5), constrained_layout=True
        )

        panels = [
            (ref - Ti, "Analytical reference", "inferno", heat_norm,
             "Temperature rise (degC)"),
            (pred - Ti, "PILNODE prediction", "inferno", heat_norm,
             "Temperature rise (degC)"),
            (absolute_error, "Absolute error", "viridis", None,
             "Absolute error (degC)"),
        ]

        for ax, (image, title, cmap, norm, label) in zip(axes, panels):
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                norm=norm,
                vmin=None if norm else 0.0,
                vmax=None if norm else error_max,
            )

            ax.set_title(f"{title}\nt = {t:g}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            if len(truth):
                ax.scatter(
                    truth[:, 0], truth[:, 1],
                    marker="x", c="cyan", s=70, linewidths=2,
                    label="Analytical sources",
                )

            if len(estimated):
                ax.scatter(
                    estimated[:, 0], estimated[:, 1],
                    marker="+", c="white", s=100, linewidths=2,
                    label="Estimated sources",
                )

            fig.colorbar(im, ax=ax, shrink=0.85, label=label)

        axes[0].legend(loc="best", fontsize=8)

        fig.suptitle(
            f"t={t:g} | RMSE={rmse:.4g} degC | "
            f"MAE={mae:.4g} degC | max error={max_error:.4g} degC"
        )

        output = out_dir / f"comparison_t{safe_time_name(t)}.png"
        fig.savefig(output, dpi=args.dpi)
        plt.close(fig)

        print(
            f"t={t:g}: RMSE={rmse:.5g}, MAE={mae:.5g}, "
            f"max error={max_error:.5g} -> {output}"
        )

    metrics_csv = out_dir / "temperature_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)

    if not metrics:
        raise RuntimeError("No valid temperature heatmaps were generated.")

    print(f"\nSource report: {source_csv.resolve()}")
    print(f"Temperature metrics: {metrics_csv.resolve()}")
    print(f"Heatmaps: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
