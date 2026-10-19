import numpy as np
import pandas as pd


def estimate_half_life(spread: pd.Series) -> float | None:
    if len(spread) < 30 or not np.isfinite(spread).all() or spread.std() < 1e-10:
        return None
    lag = spread.to_numpy()[:-1]
    _, phi = np.linalg.lstsq(np.column_stack([np.ones(len(lag)), lag]),
                             spread.to_numpy()[1:], rcond=None)[0]
    # Discrete AR(1): oscillatory/unstable roots are deliberately not assigned a half-life.
    if not 0 < phi < 1:
        return None
    result = float(-np.log(2) / np.log(phi))
    return result if np.isfinite(result) else None
