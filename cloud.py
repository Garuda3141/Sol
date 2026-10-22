from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata

GRID_SIZE = 50


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"x", "y", "t", "temperature"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The CSV contains no observations.")
    if not np.isfinite(df[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("The CSV contains non-finite values.")
    return df


def load_sources(mode: str, checkpoint_path: str | Path):
    """Return source markers as (x, y, q, label) tuples."""
    found = []

    if mode in ("simulator", "both"):
        from simulator import sources as simulator_sources
        for i, (xs, ys, q) in enumerate(simulator_sources, start=1):
            found.append((float(xs), float(ys), float(q), f"Simulator source {i}"))

    if mode in ("checkpoint", "both"):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        estimates = ckpt.get("source_estimates", ckpt.get("source_estimate"))
        if estimates is not None:
            if isinstance(estimates, dict):
                estimates = [estimates]
            for i, src in enumerate(estimates, start=1):
                xs = float(np.asarray(src["x"]).item())
                ys = float(np.asarray(src["y"]).item())
                q = float(np.asarray(src["q"]).item())
                found.append((xs, ys, q, f"Estimated source {i}"))
        else:
            # Backward compatibility with the current single-source checkpoint.
            state = ckpt.get("model_state_dict", {})
            if not all(name in state for name in ("xs", "ys", "log_q")):
                raise ValueError(
                    "Checkpoint contains no source_estimate(s) or xs/ys/log_q parameters."
                )
            cfg = ckpt.get("config", {})
            q_ref = float(cfg.get("q_ref", 1.0e4))
            xs = float(state["xs"].detach().cpu().reshape(-1)[0])
            ys = float(state["ys"].detach().cpu().reshape(-1)[0])
            log_q = float(state["log_q"].detach().cpu().reshape(-1)[0])
            q = q_ref * np.exp(log_q)
            found.append((xs, ys, float(q), "Estimated source 1"))

    return found


def draw_sources(ax, source_list):
    markers = ["X", "P", "*", "D", "s", "^", "v", "<", ">"]
    for i, (xs, ys, q, label) in enumerate(source_list):
        ax.scatter(
            [xs], [ys], marker=markers[i % len(markers)], s=170,
            edgecolors="white", linewidths=1.0, zorder=5,
            label=f"{label} (q={q:.4g})",
        )


def save_figure(fig, path: Path, show: bool):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {path}")


def plot_snapshot(df, time, output_dir, source_list, show):
    frame = df[np.isclose(df["t"], time)]
    if frame.empty:
        raise ValueError(f"No observations found at t={time}")

    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(
        frame["x"], frame["y"], c=frame["temperature"],
        cmap="inferno", s=150, edgecolors="white", linewidths=0.8,
    )
    fig.colorbar(points, ax=ax, label="Temperature (°C)")
    draw_sources(ax, source_list)
    ax.set(
        xlabel="x", ylabel="y",
        title=f"Temperature Measurements at t = {time:g}",
        xlim=(-1, 1), ylim=(-1, 1),
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    if source_list:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / f"snapshot_t_{time:g}.png", show)


def plot_interpolated_heatmap(df, time, output_dir, source_list, show):
    frame = df[np.isclose(df["t"], time)]
    if len(frame) < 3:
        raise ValueError("At least 3 sensor readings are needed for interpolation.")

    points = frame[["x", "y"]].to_numpy()
    values = frame["temperature"].to_numpy()
    gx = np.linspace(-1, 1, GRID_SIZE)
    gy = np.linspace(-1, 1, GRID_SIZE)
    X, Y = np.meshgrid(gx, gy)

    Z = griddata(points, values, (X, Y), method="linear")
    Z_nearest = griddata(points, values, (X, Y), method="nearest")
    Z = np.where(np.isnan(Z), Z_nearest, Z)

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.contourf(X, Y, Z, levels=30, cmap="inferno")
    ax.scatter(
        frame["x"], frame["y"], c=values, cmap="inferno",
        vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)),
        edgecolors="white", s=75, zorder=3,
    )
    draw_sources(ax, source_list)
    fig.colorbar(heatmap, ax=ax, label="Temperature (°C)")
    ax.set(
        xlabel="x", ylabel="y",
        title=f"Interpolated Temperature Field at t = {time:g}",
        xlim=(-1, 1), ylim=(-1, 1),
    )
    ax.set_aspect("equal")
    if source_list:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / f"heatmap_t_{time:g}.png", show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="observations.csv")
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument(
        "--sources", choices=("none", "simulator", "checkpoint", "both"),
        default="simulator",
        help="Overlay simulator truth and/or checkpoint estimates.",
    )
    parser.add_argument("--checkpoint", default="pilnode.pt")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots without opening interactive windows.")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    source_list = load_sources(args.sources, args.checkpoint)

    times = np.sort(df["t"].unique())
    print(f"Loaded {len(df)} observations")
    print(f"Times: {times}")
    print(f"Temperature range: {df['temperature'].min():.4f} to "
          f"{df['temperature'].max():.4f} °C")
    for xs, ys, q, label in source_list:
        print(f"{label}: x={xs:+.5f}, y={ys:+.5f}, q={q:.6g}")

    show = not args.no_show
    last_time = float(times[-1])
    plot_snapshot(df, last_time, output_dir, source_list, show)
    plot_interpolated_heatmap(df, last_time, output_dir, source_list, show)


if __name__ == "__main__":
    main()
