import numpy as np
import pandas as pd

from config import (
    N_SENSORS, N_TIMESTEPS, NOISE_STD, OBS_CSV, OBS_NPZ, OBS_SEED,
    SENSOR_SPREAD, SENSORS_PER_SOURCE, SOURCES, T_FIRST, TIME_MAX,
    X_MAX, X_MIN, Y_MAX, Y_MIN,
)
from simulator import Temperature

rng = np.random.default_rng(OBS_SEED)

# Clusters around each source (clipped to stay inside the domain).
xs_list, ys_list = [], []
for sx, sy, _ in SOURCES:
    xs_list.append(np.clip(rng.normal(sx, SENSOR_SPREAD, SENSORS_PER_SOURCE), X_MIN, X_MAX))
    ys_list.append(np.clip(rng.normal(sy, SENSOR_SPREAD, SENSORS_PER_SOURCE), Y_MIN, Y_MAX))

# Remaining sensors spread over the whole domain.
n_background = N_SENSORS - SENSORS_PER_SOURCE * len(SOURCES)
xs_list.append(rng.uniform(X_MIN, X_MAX, n_background))
ys_list.append(rng.uniform(Y_MIN, Y_MAX, n_background))

x = np.concatenate(xs_list)
y = np.concatenate(ys_list)
perm = rng.permutation(N_SENSORS)
x, y = x[perm], y[perm]
sensor_coords = np.column_stack((x, y))

times = np.linspace(T_FIRST, TIME_MAX, N_TIMESTEPS)

rows = []
for t in times:
    T_true = np.asarray(Temperature(x, y, float(t))).reshape(-1)
    if T_true.size != N_SENSORS or not np.all(np.isfinite(T_true)):
        raise ValueError(f"Bad temperature field at t={t:.3f}.")
    T_noisy = T_true + rng.normal(0.0, NOISE_STD, size=N_SENSORS)
    rows.extend([x[i], y[i], t, T_noisy[i], NOISE_STD] for i in range(N_SENSORS))

data = pd.DataFrame(rows, columns=["x", "y", "t", "temperature", "sigma"])
data.to_csv(OBS_CSV, index=False)
np.savez_compressed(
    OBS_NPZ,
    coords=data[["x", "y", "t"]].to_numpy(dtype=np.float64),
    targets=data[["temperature"]].to_numpy(dtype=np.float64),
    sigma=np.float64(NOISE_STD),
    sensor_positions=sensor_coords,
    times=times,
)

print(f"Saved {len(data)} observations  ({N_SENSORS} sensors x {N_TIMESTEPS} times, "
      f"{len(SOURCES)} sources, noise std {NOISE_STD}).")
print(f"Temperature range: {data['temperature'].min():.3f} to {data['temperature'].max():.3f} degC")
