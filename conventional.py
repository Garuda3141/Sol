
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import exp1

from config import (
    OBS_NPZ, SOURCES, ALPHA, K_COND, SOURCE_RADIUS, TI,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
)

# 1. Load observations
with np.load(OBS_NPZ) as data:
    coords = data["coords"]
    T_meas = data["targets"].ravel()
    sigma = float(data["sigma"])
    sensor_positions = data["sensor_positions"]
    times = data["times"]

x, y, t = coords.T
K = len(SOURCES)  # Number of sources; do not use their true parameters.

# 2. Parameterized analytical forward model
def forward(params):
    T = np.full(len(t), TI, dtype=float)
    valid = t > 0

    for j in range(K):
        xs, ys, q = params[3*j:3*j+3]

        r2 = (x - xs)**2 + (y - ys)**2 + SOURCE_RADIUS**2
        z = r2[valid] / (4.0 * ALPHA * t[valid])

        T[valid] += (
            q / (4.0 * np.pi * K_COND)
        ) * exp1(z)

    return T

# 3. Weighted residuals
def residuals(params):
    return (forward(params) - T_meas) / sigma

# 4. Initialize source locations using high-temperature sensors
n_sensors = len(sensor_positions)
temperature_matrix = T_meas.reshape(len(times), n_sensors)
mean_temperature = temperature_matrix.mean(axis=0)

# Select spatially separated hot sensors as initial source guesses.
order = np.argsort(mean_temperature)[::-1]
chosen = []

min_distance = 0.15 * np.hypot(X_MAX - X_MIN, Y_MAX - Y_MIN)

for idx in order:
    point = sensor_positions[idx]

    if all(np.linalg.norm(point - sensor_positions[j]) >= min_distance
           for j in chosen):
        chosen.append(idx)

    if len(chosen) == K:
        break

# Fallback if the sensor positions are too close together.
for idx in order:
    if len(chosen) == K:
        break
    if idx not in chosen:
        chosen.append(idx)

p0 = []
for idx in chosen:
    p0.extend([*sensor_positions[idx], 1.0])

p0 = np.asarray(p0, dtype=float)

# Bound source locations to the domain; allow signed source strengths.
lower = np.tile([X_MIN, Y_MIN, -np.inf], K)
upper = np.tile([X_MAX, Y_MAX, np.inf], K)

# 5. Solve the inverse problem
result = least_squares(
    residuals,
    p0,
    bounds=(lower, upper),
    method="trf",
    jac="2-point",
    x_scale="jac",
    max_nfev=5000,
)

T_fit = forward(result.x)
error = T_meas - T_fit

print("Converged:", result.success)
print("Message:", result.message)
print("Weighted least-squares cost:", result.cost)
print("RMSE:", np.sqrt(np.mean(error**2)), "°C")
print("MAE:", np.mean(np.abs(error)), "°C")

for j in range(K):
    print(
        f"Source {j+1}: "
        f"x={result.x[3*j]:.4f}, "
        f"y={result.x[3*j+1]:.4f}, "
        f"q={result.x[3*j+2]:.4f}"
    )

# 6. Plot measured versus fitted temperatures and residuals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(T_meas, T_fit, s=12, alpha=0.5)
lo = min(T_meas.min(), T_fit.min())
hi = max(T_meas.max(), T_fit.max())
axes[0].plot([lo, hi], [lo, hi], "k--")
axes[0].set(
    xlabel="Measured temperature (°C)",
    ylabel="Fitted temperature (°C)",
    title="Measured vs. fitted",
)

axes[1].scatter(T_fit, error, s=12, alpha=0.5)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set(
    xlabel="Fitted temperature (°C)",
    ylabel="Measured - fitted (°C)",
    title="Residuals",
)

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Compare spatial measurements and predictions at one time
selected_time = times[len(times) // 2]
mask = np.isclose(t, selected_time)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, values, title in [
    (axes[0], T_meas, "Measured field"),
    (axes[1], T_fit, "Fitted field"),
]:
    sc = ax.scatter(x[mask], y[mask], c=values[mask],
                    s=65, cmap="inferno")
    ax.set_title(f"{title}, t={selected_time:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="Temperature (°C)")

plt.tight_layout()
plt.show()
