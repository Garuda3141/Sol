
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg" if False else "TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

import config as C
from modelviz import (
    load_model,
    analytical_temperature,
    normalize_sources,
)
from simulator import sources as DEFAULT_SOURCES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate PILNODE versus analytical heat diffusion."
    )
    parser.add_argument("--checkpoint", default=C.CHECKPOINT)
    parser.add_argument("--out-dir", default=C.VIZ_DIR)
    parser.add_argument("--grid", type=int, default=C.VIZ_GRID)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--xmin", type=float, default=C.X_MIN)
    parser.add_argument("--xmax", type=float, default=C.X_MAX)
    parser.add_argument("--ymin", type=float, default=C.Y_MIN)
    parser.add_argument("--ymax", type=float, default=C.Y_MAX)
    parser.add_argument("--vmin", type=float, default=20.0)
    parser.add_argument("--vmax", type=float, default=100.0)
    parser.add_argument("--error-vmax", type=float, default=None)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    parser.add_argument("--save", choices=["none", "gif", "mp4"],
                        default="gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--truth-sources", nargs="+", type=float, default=None,
        help="Optional flat sequence of x y q triples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.grid < 2 or args.frames < 1:
        raise ValueError("--grid must be >= 2 and --frames must be >= 1.")
    if args.interval < 1 or args.fps < 1:
        raise ValueError("--interval and --fps must be positive.")
    if not args.vmin < args.vmax:
        raise ValueError("--vmin must be smaller than --vmax.")

    device = torch.device(args.device)
    model, cfg, _ = load_model(Path(args.checkpoint), device)

    truth = normalize_sources(
        DEFAULT_SOURCES if args.truth_sources is None
        else args.truth_sources,
        "truth sources",
    )

    estimated_dict = model.source_estimate()
    estimated = np.column_stack([
        np.asarray(estimated_dict[name]).reshape(-1)
        for name in ("x", "y", "q")
    ])

    x = np.linspace(args.xmin, args.xmax, args.grid)
    y = np.linspace(args.ymin, args.ymax, args.grid)
    X, Y = np.meshgrid(x, y)

    # Avoid t=0 because the trained model uses a positive time domain.
    times = np.linspace(
        max(cfg.t_max / args.frames, cfg.t_max * 1e-6),
        cfg.t_max,
        args.frames,
    )

    xt = torch.as_tensor(
        X.reshape(-1, 1), dtype=torch.float32, device=device
    )
    yt = torch.as_tensor(
        Y.reshape(-1, 1), dtype=torch.float32, device=device
    )

    # Precompute fields so animation updates only change image data.
    predictions = []
    references = []
    errors = []
    rmses = []
    maes = []

    with torch.inference_mode():
        for i, t in enumerate(times):
            tt = torch.full_like(xt, float(t))
            pred = model(xt, yt, tt).cpu().numpy().reshape(X.shape)

            ref = analytical_temperature(X, Y, float(t), truth)

            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite prediction at t={t:g}.")

            error = np.abs(pred - ref)

            predictions.append(pred)
            references.append(ref)
            errors.append(error)
            rmses.append(float(np.sqrt(np.mean((pred - ref) ** 2))))
            maes.append(float(np.mean(error)))

            if (i + 1) % 10 == 0 or i + 1 == len(times):
                print(f"Precomputed {i + 1}/{len(times)} frames")

    predictions = np.asarray(predictions)
    references = np.asarray(references)
    errors = np.asarray(errors)

    # Shared normalization keeps all frames directly comparable.
    error_vmax = (
        args.error_vmax
        if args.error_vmax is not None
        else max(float(errors.max()), 1e-8)
    )
    if error_vmax <= 0:
        raise ValueError("--error-vmax must be positive.")

    extent = [args.xmin, args.xmax, args.ymin, args.ymax]
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 6), constrained_layout=True
    )

    fields = [
        references[0],
        predictions[0],
        errors[0],
    ]
    titles = [
        "Analytical reference",
        "PILNODE prediction",
        "Absolute error",
    ]
    cmaps = ["inferno", "inferno", "viridis"]
    limits = [
        (args.vmin, args.vmax),
        (args.vmin, args.vmax),
        (0.0, error_vmax),
    ]

    images = []
    for ax, field, title, cmap, (vmin, vmax) in zip(
        axes, fields, titles, cmaps, limits
    ):
        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.grid(alpha=0.15)
        images.append(im)

    # Distinguish true source positions from estimated positions.
    for ax in axes:
        if len(truth):
            ax.scatter(
                truth[:, 0], truth[:, 1],
                marker="*", s=130, c="cyan",
                edgecolors="white", linewidths=0.7,
                label="True sources", zorder=5,
            )
        if len(estimated):
            ax.scatter(
                estimated[:, 0], estimated[:, 1],
                marker="+", s=110, c="white",
                linewidths=2, label="Estimated sources",
                zorder=6,
            )

    axes[0].legend(loc="upper right", fontsize=8)
    suptitle = fig.suptitle("Preparing animation")

    def update(frame):
        t = times[frame]

        images[0].set_data(references[frame])
        images[1].set_data(predictions[frame])
        images[2].set_data(errors[frame])

        suptitle.set_text(
            f"Heat diffusion | t={t:.5g} | "
            f"RMSE={rmses[frame]:.4g} °C | "
            f"MAE={maes[frame]:.4g} °C"
        )

        return (*images, suptitle)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save == "gif":
        output = out_dir / "pilnode_animation.gif"
        ani.save(
            output,
            writer=animation.PillowWriter(fps=args.fps),
            dpi=args.dpi,
        )
        print(f"Saved GIF: {output.resolve()}")

    elif args.save == "mp4":
        output = out_dir / "pilnode_animation.mp4"
        ani.save(
            output,
            writer=animation.FFMpegWriter(fps=args.fps),
            dpi=args.dpi,
        )
        print(f"Saved MP4: {output.resolve()}")

    else:
        # TkAgg supports an interactive window on a desktop.
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
