import math

# =============================================================================
# 1. PHYSICS (analytical simulator)
# =============================================================================
K_COND = 15.0            # thermal conductivity            (W/m/K)
RHO = 8000.0             # density                         (kg/m^3)
CP = 50.0                # specific heat                   (J/kg/K)
ALPHA = K_COND / (RHO * CP)   # thermal diffusivity        (m^2/s) = 3.75e-5
TI = 20.0                # uniform initial temperature     (degC)
SOURCE_RADIUS = 0.01     # effective source radius         (m)

SOURCES = [
    (-0.30, -0.25, 7000.0),
    (+0.30, +0.25, 5000.0),
]

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
TIME_MAX = 500.0         # last measurement time           (s)


DIFFUSION_LENGTH = math.sqrt(4.0 * ALPHA * TIME_MAX)

OBS_SEED = 42
NOISE_STD = 0.05         # sensor noise std                (degC)
N_SENSORS = 30
N_TIMESTEPS = 100
T_FIRST = 1.0            # first measurement time          (s)
SENSORS_PER_SOURCE = 10  # sensors clustered around EACH source; the rest are uniform background
SENSOR_SPREAD = 0.12     # gaussian std of those clusters  (m)
OBS_CSV = "observations.csv"
OBS_NPZ = "observations.npz"


N_SOURCES = len(SOURCES)     # K the model tries to recover; override with --n-sources
LATENT_DIM = 8
DYN_HIDDEN = 64
HIDDEN = 128
DEPTH = 3
ODE_STEPS = 8                # RK4 steps over the latent trajectory
TEMP_SCALE = 100.0           # degC; the network predicts (T - TI) / TEMP_SCALE
Q_REF = 1.0e4                # q = Q_REF * exp(log_q)
SOURCE_FEATURES = True       # feed source-centred (dx, dy, ln r^2) features to the decoder


SEED = 42
EPOCHS = 1500
BATCH_SIZE = 128             # ~3000 observations -> ~24 steps/epoch
LR_NET = 1.0e-4
LR_SOURCE = 1.0e-2
LAMBDA_PDE = 1.0
SIGMA_FLOOR = 0.5            # degC; data residuals divided by max(sigma, floor)
GRAD_CLIP = 1.0
LOG_EVERY = 100
LR_FLOOR = 0.01              # cosine decay ends at this fraction of the base lr

# Collocation sampling (model.sample_collocation)
N_COLLOC = 128 * N_SOURCES   # points per step; near-source points are split over the K sources
NEAR_FRAC = 0.5              # fraction of points placed around the current source estimates
R_MIN = 0.005                # m, log-spaced radius range around each source
R_MAX = 0.5

# Source initial guess (None -> data-driven weighted k-means)
XS0 = None
YS0 = None
Q0 = [1.0e4]                 # one value (broadcast) or exactly K values
INIT_POWER = 1.0             # >1 sharpens the init towards the hottest sensors

# Output files
CHECKPOINT = "pilnode.pt"
HISTORY_CSV = "training_history.csv"


VIZ_DIR = "visualizations"
VIZ_GRID = 201
VIZ_DPI = 160
VIZ_TIMES = None             # None -> 7 geometric fractions of the checkpoint t_max
VIZ_HEAT_GAMMA = 0.4         # PowerNorm exponent for the temperature-rise maps
VIZ_ERROR_VMAX = None        # fixed colour-scale ceiling for the error maps (degC), or None

assert N_SENSORS >= SENSORS_PER_SOURCE * len(SOURCES), "not enough sensors for the per-source clusters"
assert all(X_MIN < sx < X_MAX and Y_MIN < sy < Y_MAX for sx, sy, _ in SOURCES), "sources must lie inside the domain"
assert TIME_MAX > T_FIRST
