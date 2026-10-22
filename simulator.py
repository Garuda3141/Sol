import numpy as np
import scipy.special

from config import (
    ALPHA as alpha,
    CP as cp,
    K_COND as k,
    RHO as rho,
    SOURCE_RADIUS as source_radius,
    SOURCES as sources,
    TI as Ti,
)


def Temperature(X, Y, t):
    """T(x, y, t) = Ti + sum_j q_j / (4 pi k) * E1((r_j^2 + a^2) / (4 alpha t))."""
    T = np.full(np.broadcast_shapes(np.shape(X), np.shape(Y)), Ti, dtype=float)
    if t <= 0:
        return T
    for xs, ys, q in sources:
        R2 = (X - xs) ** 2 + (Y - ys) ** 2 + source_radius ** 2
        T += (q / (4.0 * np.pi * k)) * scipy.special.exp1(R2 / (4.0 * alpha * t))
    return T
