import numpy as np

HCAR_PERIODS = (4, 2, 1)


def _extract_channel_state(series_2d: np.ndarray) -> np.ndarray:
    series_2d = series_2d.astype(np.float32)
    mean_val = series_2d.mean(axis=0)
    std_val = series_2d.std(axis=0)
    if series_2d.shape[0] > 1:
        slope_val = (series_2d[-1] - series_2d[0]) / float(series_2d.shape[0] - 1)
        abs_diff = np.mean(np.abs(series_2d[1:] - series_2d[:-1]), axis=0)
    else:
        slope_val = np.zeros_like(mean_val, dtype=np.float32)
        abs_diff = np.zeros_like(mean_val, dtype=np.float32)
    return np.stack([mean_val, std_val, slope_val, abs_diff], axis=-1).astype(np.float32)


def _decompose_period(seq_x: np.ndarray, period: int) -> np.ndarray:
    seq_x = seq_x.astype(np.float32)
    if period <= 1:
        cur = seq_x
    else:
        cut = (seq_x.shape[0] // period) * period
        if cut <= 0:
            cur = seq_x
        else:
            cur = seq_x[:cut].reshape(-1, period, seq_x.shape[1]).mean(axis=1)
            cur = np.repeat(cur, period, axis=0)
    return (cur - cur[-1:]).astype(np.float32)


def _build_periodic_local_state(seq_x: np.ndarray) -> np.ndarray:
    states = []
    for period in HCAR_PERIODS:
        period_series = _decompose_period(seq_x, period)
        channel_state = _extract_channel_state(period_series)  # [C, 4]
        states.append(channel_state)
    return np.stack(states, axis=0).astype(np.float32)  # [G, C, 4]


def build_meta_record(seq_x: np.ndarray):
    return {
        "local_state_by_period": _build_periodic_local_state(seq_x),
    }
