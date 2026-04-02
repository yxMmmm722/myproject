import os

import numpy as np

HCAR_PERIODS = (4, 2, 1)


def infer_dataset_name(args, data_path: str) -> str:
    if hasattr(args, "data") and args.data != "custom":
        return str(args.data)
    return os.path.splitext(os.path.basename(data_path))[0]


def _compute_slope(series: np.ndarray) -> float:
    if series.size <= 1:
        return 0.0
    t = np.arange(series.size, dtype=np.float32)
    t = t - t.mean()
    y = series.astype(np.float32) - series.astype(np.float32).mean()
    denom = float((t * t).sum()) + 1e-6
    return float((t * y).sum() / denom)


def _compute_stats(series: np.ndarray) -> np.ndarray:
    series = series.astype(np.float32)
    mean_val = float(series.mean())
    std_val = float(series.std())
    centered = series - mean_val
    skew_val = float((centered ** 3).mean() / ((std_val + 1e-6) ** 3))
    slope_val = _compute_slope(series)
    return np.array([mean_val, std_val, skew_val, slope_val], dtype=np.float32)


def _downsample_non_overlap(series: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return series
    cut = (series.size // period) * period
    if cut <= 0:
        return series
    reshaped = series[:cut].reshape(-1, period)
    return reshaped.mean(axis=1)


def _build_periodic_local_state(seq_x: np.ndarray) -> np.ndarray:
    signal = seq_x.astype(np.float32).mean(axis=1)
    signal = signal - signal[-1]
    states = []
    for period in HCAR_PERIODS:
        ds_signal = _downsample_non_overlap(signal, period)
        states.append(_compute_stats(ds_signal))
    return np.stack(states, axis=0)


def build_meta_record(timestamp, dataset_name: str, target: str, feature_names, seq_x: np.ndarray):
    del timestamp, dataset_name, target, feature_names
    return {
        "local_state_by_period": _build_periodic_local_state(seq_x).astype(np.float32),
    }
