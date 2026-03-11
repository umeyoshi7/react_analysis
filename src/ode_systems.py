"""ODE system definitions and solver utilities for reaction kinetics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# ODE definitions
# ---------------------------------------------------------------------------

def _ode_simple(t, y, k, n):
    A = max(y[0], 0.0)
    return [-k * A ** n]


def _ode_sequential(t, y, k1, k2, n1, n2):
    A = max(y[0], 0.0)
    B = max(y[1], 0.0)
    # When concentration is 0, rate is 0 regardless of order (avoids 0**0 = 1)
    rate1 = k1 * A**n1 if A > 0.0 else 0.0
    rate2 = k2 * B**n2 if B > 0.0 else 0.0
    return [-rate1, rate1 - rate2, rate2]


def _ode_parallel(t, y, k1, k2, n):
    A = max(y[0], 0.0)
    # When concentration is 0, rate is 0 regardless of order
    rate = (k1 + k2) * A**n if A > 0.0 else 0.0
    r1 = k1 / (k1 + k2) * rate
    r2 = k2 / (k1 + k2) * rate
    return [-rate, r1, r2]


# ---------------------------------------------------------------------------
# Solver helper
# ---------------------------------------------------------------------------

def solve_and_predict(
    ode_fn,
    t_span: tuple[float, float],
    y0: list[float],
    t_eval: np.ndarray,
    args: tuple,
) -> np.ndarray | None:
    """Run solve_ivp; returns sol.y or None on failure."""
    try:
        sol = solve_ivp(
            ode_fn, t_span, y0,
            method="RK45", t_eval=t_eval, args=args,
            rtol=1e-6, atol=1e-9, dense_output=False,
        )
        if sol.success and sol.y.shape[1] == len(t_eval):
            return sol.y
    except Exception:
        pass
    return None
