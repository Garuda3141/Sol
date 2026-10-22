import simulator as sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# --------------------------------------------------
# 4. Spatial grid
# --------------------------------------------------
x = np.linspace(-0.75, 0.75, 350)
y = np.linspace(-0.75, 0.75, 350)
X, Y = np.meshgrid(x, y)

# --------------------------------------------------
# 5. Figure and fixed temperature scale
# --------------------------------------------------
vmin = 20.0
vmax = 100.0
levels = np.linspace(vmin, vmax, 65)

fig, ax = plt.subplots(figsize=(10, 8))

fig.subplots_adjust(
    left=0.10, right=0.85,
    bottom=0.10, top=0.90
)

sm = plt.cm.ScalarMappable(
    cmap="inferno",
    norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
sm.set_array([])

cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.70])
fig.colorbar(sm, cax=cbar_ax, label="Temperature (°C)")

# --------------------------------------------------
# 6. Animation update
# --------------------------------------------------
def update(frame):
    ax.clear()

    current_time = frame * 100.0
    T_field = sim.Temperature(X, Y, current_time)

    ax.contourf(
        X, Y, T_field,
        levels=levels,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        extend="max"
    )

    # Draw source positions
    for i, (xs, ys, q) in enumerate(sim.sources, start=1):
        ax.scatter(
            xs, ys,
            marker="*",
            s=150,
            c="cyan",
            edgecolors="white",
            linewidths=0.8,
            zorder=5
        )
        ax.annotate(
            f"S{i}: {q:.0f}",
            (xs, ys),
            xytext=(7, 8),
            textcoords="offset points",
            color="white",
            fontsize=8,
            fontweight="bold"
        )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_title(
        f"Multi-Source Heat Diffusion | t = {current_time:.0f} s"
    )
    ax.grid(alpha=0.15)

    return []

# --------------------------------------------------
# 7. Run animation
# --------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=20,
    interval=100,
    blit=False,
    repeat=True
)

plt.show()
