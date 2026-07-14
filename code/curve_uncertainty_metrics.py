"""Curve-level prediction-interval metrics used in the PBPF response-curve analysis.

This module implements the calibrated residual-quantile interval construction and
per-curve PICP, interval-width (IW), and interval-score (IS) calculations used
for DSC heating, DSC cooling, and UTM stress-strain response curves.
"""
from __future__ import annotations

import numpy as np


def calibrated_residual_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct pointwise residual-quantile prediction intervals.

    Parameters
    ----------
    y_true, y_pred
        Arrays with shape ``(n_samples, n_curve_points)``.
    alpha
        Miscoverage level; ``0.20`` defines a nominal 80% interval.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be two-dimensional arrays with identical shapes.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    residual = y_true - y_pred
    residual_low = np.quantile(residual, alpha / 2.0, axis=0)
    residual_high = np.quantile(residual, 1.0 - alpha / 2.0, axis=0)
    lower = y_pred + residual_low[None, :]
    upper = y_pred + residual_high[None, :]
    return np.minimum(lower, upper), np.maximum(lower, upper)


def curve_interval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.20,
) -> dict[str, np.ndarray | float]:
    """Calculate curve-point PICP, IW, and IS for calibrated intervals.

    The interval score is ``width + (2 / alpha) * outside_distance``. Returned
    sample-level metrics are averaged over curve points; ``*_mean`` values are
    then averaged over samples.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower, upper = calibrated_residual_intervals(y_true, y_pred, alpha=alpha)
    width = upper - lower
    inside = (y_true >= lower) & (y_true <= upper)
    outside_distance = np.maximum(lower - y_true, 0.0) + np.maximum(y_true - upper, 0.0)
    interval_score = width + (2.0 / alpha) * outside_distance

    picp = inside.mean(axis=1)
    iw = width.mean(axis=1)
    iscore = interval_score.mean(axis=1)
    return {
        "lower": lower,
        "upper": upper,
        "picp_per_curve": picp,
        "iw_per_curve": iw,
        "is_per_curve": iscore,
        "picp_mean": float(picp.mean()),
        "iw_mean": float(iw.mean()),
        "is_mean": float(iscore.mean()),
    }
