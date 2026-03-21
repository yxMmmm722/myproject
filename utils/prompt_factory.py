import hashlib
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HCAR_PERIODS = (4, 2, 1)


def infer_dataset_name(args, data_path: str) -> str:
    if hasattr(args, "data") and args.data != "custom":
        return str(args.data)
    return os.path.splitext(os.path.basename(data_path))[0]


def _safe_hash(text: str, modulo: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def _dataset_family(dataset_name: str) -> str:
    name = dataset_name.lower()
    if "etth" in name or "ettm" in name or name == "ett":
        return "ett"
    if "electricity" in name:
        return "electricity"
    if "exchange" in name:
        return "exchange_rate"
    if "illness" in name or "ili" in name:
        return "illness"
    if "traffic" in name:
        return "traffic"
    if "weather" in name:
        return "weather"
    return "generic"


def _cyclic_encode(value: int, period: int) -> Tuple[float, float]:
    angle = 2.0 * np.pi * float(value) / float(period)
    return float(np.sin(angle)), float(np.cos(angle))


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
    # Channel-mean signal with offset removal for context-side statistics.
    signal = seq_x.astype(np.float32).mean(axis=1)
    signal = signal - signal[-1]
    states = []
    for p in HCAR_PERIODS:
        ds_signal = _downsample_non_overlap(signal, p)
        states.append(_compute_stats(ds_signal))
    return np.stack(states, axis=0)  # [G, 4]


def _trend_descriptor(slope: float, std: float) -> str:
    if slope > 0.02:
        return "upward"
    if slope < -0.02:
        return "downward"
    if std > 0.8:
        return "volatile"
    return "stable"


def _volatility_descriptor(std: float) -> str:
    if std > 1.2:
        return "high"
    if std > 0.6:
        return "medium"
    return "low"


def _season_from_month(month: int) -> Tuple[int, str]:
    # 0:spring, 1:summer, 2:autumn, 3:winter
    if month in (3, 4, 5):
        return 0, "spring"
    if month in (6, 7, 8):
        return 1, "summer"
    if month in (9, 10, 11):
        return 2, "autumn"
    return 3, "winter"


def _entity_name(feature_names, target: str) -> str:
    if target and target in feature_names:
        return str(target)
    if feature_names:
        return str(feature_names[-1])
    return "target"


def _time_and_state_features(ts: pd.Timestamp, local_state_by_period: np.ndarray):
    day_of_week = int(ts.dayofweek)
    hour = int(ts.hour)
    month = int(ts.month)
    week_of_year = int(ts.isocalendar().week)
    is_holiday = int(day_of_week >= 5)
    weekday_name = ts.day_name()

    hour_sin, hour_cos = _cyclic_encode(hour, 24)
    weekday_sin, weekday_cos = _cyclic_encode(day_of_week, 7)
    month_sin, month_cos = _cyclic_encode(month, 12)
    week_sin, week_cos = _cyclic_encode(week_of_year, 53)

    window_mean, window_std, window_skew, window_slope = local_state_by_period[-1].tolist()
    trend = _trend_descriptor(window_slope, window_std)
    volatility = _volatility_descriptor(window_std)
    season_id, season_name = _season_from_month(month)

    return {
        "day_of_week": day_of_week,
        "hour": hour,
        "month": month,
        "week_of_year": week_of_year,
        "is_holiday": is_holiday,
        "weekday_name": weekday_name,
        "time_features": np.array(
            [hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin, month_cos, week_sin, week_cos],
            dtype=np.float32,
        ),
        "window_mean": float(window_mean),
        "window_std": float(window_std),
        "window_skewness": float(window_skew),
        "trend_indicator": float(window_slope),
        "trend_descriptor": trend,
        "volatility_descriptor": volatility,
        "season_id": season_id,
        "season_name": season_name,
    }


def _state_descriptor_from_stats(stats: np.ndarray) -> Dict[str, float]:
    mean_val, std_val, skew_val, slope_val = [float(x) for x in stats.tolist()]
    return {
        "mean": mean_val,
        "std": std_val,
        "skew": skew_val,
        "slope": slope_val,
        "trend": _trend_descriptor(slope_val, std_val),
        "volatility": _volatility_descriptor(std_val),
    }


def _build_ett_meta(base: Dict) -> Dict:
    hour = base["hour"]
    if hour in {7, 8, 9, 17, 18, 19}:
        load_phase = "peak-load"
        load_phase_id = 2
    elif hour in {10, 11, 12, 13, 14, 15, 16}:
        load_phase = "mid-load"
        load_phase_id = 1
    else:
        load_phase = "off-load"
        load_phase_id = 0

    return {
        "sensor_type": "oil_transformer_sensor",
        "physical_location": f"transformer_{base['dataset_name'].lower()}",
        "peak_status_id": int(load_phase_id == 2),
        "load_phase": load_phase,
        "load_phase_id": load_phase_id,
        "exogenous_vars": np.array(
            [float(base["is_holiday"]), float(load_phase_id), float(base["season_id"]), float(base["week_of_year"]) / 53.0],
            dtype=np.float32,
        ),
    }


def _build_electricity_meta(base: Dict) -> Dict:
    hour = base["hour"]
    if hour in {0, 1, 2, 3, 4, 5, 6, 23}:
        tariff_name = "valley"
        tariff_id = 0
    elif hour in {7, 8, 9, 17, 18, 19, 20}:
        tariff_name = "peak"
        tariff_id = 2
    else:
        tariff_name = "flat"
        tariff_id = 1

    return {
        "sensor_type": "smart_meter_cluster",
        "physical_location": f"grid_zone_{base['dataset_name'].lower()}",
        "peak_status_id": int(tariff_id == 2),
        "tariff_name": tariff_name,
        "tariff_id": tariff_id,
        "exogenous_vars": np.array(
            [float(base["is_holiday"]), float(tariff_id), float(base["season_id"]), float(base["hour"]) / 23.0],
            dtype=np.float32,
        ),
    }


def _build_exchange_meta(base: Dict) -> Dict:
    hour = base["hour"]
    if hour < 8:
        session = "asia"
        session_id = 0
    elif hour < 16:
        session = "europe"
        session_id = 1
    else:
        session = "america"
        session_id = 2

    month_end_flag = int(pd.Timestamp(base["timestamp"]).is_month_end)
    return {
        "sensor_type": "fx_pair_bundle",
        "physical_location": "global_fx_market",
        "peak_status_id": int(session_id >= 1),
        "trading_session": session,
        "trading_session_id": session_id,
        "month_end_flag": month_end_flag,
        "exogenous_vars": np.array(
            [float(base["is_holiday"]), float(month_end_flag), float(session_id), float(base["season_id"])],
            dtype=np.float32,
        ),
    }


def _build_illness_meta(base: Dict) -> Dict:
    flu_season_flag = int(base["month"] in {11, 12, 1, 2, 3})
    return {
        "sensor_type": "epi_surveillance_stream",
        "physical_location": "public_health_region",
        "peak_status_id": flu_season_flag,
        "epi_week": int(base["week_of_year"]),
        "flu_season_flag": flu_season_flag,
        "exogenous_vars": np.array(
            [float(base["is_holiday"]), float(flu_season_flag), float(base["season_id"]), float(base["week_of_year"]) / 53.0],
            dtype=np.float32,
        ),
    }


def _build_traffic_meta(base: Dict) -> Dict:
    hour = base["hour"]
    if hour in {7, 8, 9, 17, 18, 19}:
        traffic_regime = "rush-hour"
        regime_id = 2
    elif hour in {6, 10, 16, 20, 21}:
        traffic_regime = "transition"
        regime_id = 1
    else:
        traffic_regime = "free-flow"
        regime_id = 0

    return {
        "sensor_type": "road_loop_detector",
        "physical_location": f"urban_road_{base['dataset_name'].lower()}",
        "peak_status_id": int(regime_id == 2),
        "traffic_regime": traffic_regime,
        "traffic_regime_id": regime_id,
        "exogenous_vars": np.array(
            [float(base["is_holiday"]), float(regime_id), float(base["season_id"]), float(base["hour"]) / 23.0],
            dtype=np.float32,
        ),
    }


def _build_weather_meta(base: Dict) -> Dict:
    daylight_flag = int(6 <= base["hour"] < 18)
    return {
        "sensor_type": "meteorological_station",
        "physical_location": f"station_{base['dataset_name'].lower()}",
        "peak_status_id": daylight_flag,
        "daylight_flag": daylight_flag,
        "exogenous_vars": np.array(
            [float(base["is_holiday"]), float(daylight_flag), float(base["season_id"]), float(base["hour"]) / 23.0],
            dtype=np.float32,
        ),
    }


def _build_generic_meta(base: Dict) -> Dict:
    peak_status_id = int(base["hour"] in {7, 8, 9, 17, 18, 19})
    return {
        "sensor_type": "generic_sensor",
        "physical_location": f"{base['dataset_name'].lower()}_entity",
        "peak_status_id": peak_status_id,
        "exogenous_vars": np.array(
            [float(base["is_holiday"]), float(peak_status_id), float(base["season_id"]), float(base["hour"]) / 23.0],
            dtype=np.float32,
        ),
    }


class PromptFactory:
    def __init__(self):
        self.template_catalog = {
            "ett": "ETT node {physical_location}. Time {hour}:00 on {weekday_name}. Load phase={load_phase}; state={trend_descriptor}/{volatility_descriptor}.",
            "electricity": "Electricity grid {physical_location}. {weekday_name} {hour}:00, tariff={tariff_name}, holiday={is_holiday}. Demand state={trend_descriptor}/{volatility_descriptor}.",
            "exchange_rate": "FX market at {trading_session} session, weekday={weekday_name}, month_end={month_end_flag}. Price state={trend_descriptor}/{volatility_descriptor}.",
            "illness": "Illness stream {physical_location}, epi-week={epi_week}, season={season_name}, flu_season={flu_season_flag}. Incidence state={trend_descriptor}/{volatility_descriptor}.",
            "traffic": "Traffic group {physical_location}, regime={traffic_regime}, time={hour}:00 {weekday_name}. Congestion state={trend_descriptor}/{volatility_descriptor}.",
            "weather": "Weather station {physical_location}, season={season_name}, daylight={daylight_flag}, time={hour}:00. Atmosphere state={trend_descriptor}/{volatility_descriptor}.",
            "generic": "Dataset {dataset_name} at {physical_location}. Time={hour}:00 {weekday_name}, holiday={is_holiday}. State={trend_descriptor}.",
            "periodic_suffix": "Scale={period}; mean={mean:.4f}, std={std:.4f}, skew={skew:.4f}, slope={slope:.4f}, trend={trend}, volatility={volatility}.",
        }

    def get_template_catalog(self) -> Dict[str, str]:
        return dict(self.template_catalog)

    def build(self, meta_data: Dict) -> str:
        family = meta_data["dataset_family"]

        if family == "ett":
            return (
                f"ETT node {meta_data['physical_location']}. "
                f"Time {meta_data['hour']:02d}:00 on {meta_data['weekday_name']}. "
                f"Load phase is {meta_data['load_phase']}; state is {meta_data['trend_descriptor']} trend "
                f"with {meta_data['volatility_descriptor']} volatility."
            )

        if family == "electricity":
            return (
                f"Electricity grid {meta_data['physical_location']}. "
                f"{meta_data['weekday_name']} {meta_data['hour']:02d}:00, tariff={meta_data['tariff_name']}, "
                f"holiday={meta_data['is_holiday']}. Demand shows {meta_data['trend_descriptor']} trend "
                f"with {meta_data['volatility_descriptor']} variance."
            )

        if family == "exchange_rate":
            return (
                f"FX market context at {meta_data['trading_session']} session, "
                f"weekday={meta_data['weekday_name']}, month_end={meta_data['month_end_flag']}. "
                f"Price dynamics indicate {meta_data['volatility_descriptor']} volatility and "
                f"{meta_data['trend_descriptor']} trend."
            )

        if family == "illness":
            return (
                f"Illness surveillance stream {meta_data['physical_location']}, epi-week {meta_data['epi_week']}, "
                f"season={meta_data['season_name']}, flu_season={meta_data['flu_season_flag']}. "
                f"Incidence exhibits {meta_data['trend_descriptor']} trend with "
                f"{meta_data['volatility_descriptor']} fluctuation."
            )

        if family == "traffic":
            return (
                f"Traffic sensor group {meta_data['physical_location']} during {meta_data['traffic_regime']} "
                f"at {meta_data['hour']:02d}:00 on {meta_data['weekday_name']}. "
                f"Congestion signal is {meta_data['trend_descriptor']} with "
                f"{meta_data['volatility_descriptor']} volatility."
            )

        if family == "weather":
            return (
                f"Weather station {meta_data['physical_location']} at {meta_data['hour']:02d}:00, "
                f"season={meta_data['season_name']}, daylight={meta_data['daylight_flag']}. "
                f"Atmospheric state is {meta_data['trend_descriptor']} with "
                f"{meta_data['volatility_descriptor']} variance."
            )

        return (
            f"Dataset {meta_data['dataset_name']} at {meta_data['physical_location']}. "
            f"Time: {meta_data['hour']:02d}:00 on {meta_data['weekday_name']} "
            f"(holiday={meta_data['is_holiday']}). "
            f"State: {meta_data['trend_descriptor']} trend with {meta_data['window_std']:.4f} variance."
        )

    def build_by_period(self, meta_data: Dict, local_state_by_period: np.ndarray) -> List[str]:
        base_text = self.build(meta_data)
        texts = []
        for period, stats in zip(HCAR_PERIODS, local_state_by_period):
            desc = _state_descriptor_from_stats(np.asarray(stats, dtype=np.float32))
            suffix = (
                f"Scale={period}; mean={desc['mean']:.4f}, std={desc['std']:.4f}, "
                f"skew={desc['skew']:.4f}, slope={desc['slope']:.4f}, "
                f"trend={desc['trend']}, volatility={desc['volatility']}."
            )
            texts.append(f"{base_text} {suffix}")
        return texts


def build_meta_record(
    timestamp,
    dataset_name: str,
    target: str,
    feature_names,
    seq_x: np.ndarray,
    prompt_factory: PromptFactory,
) -> Tuple[Dict, str, List[str]]:
    ts = pd.Timestamp(timestamp)
    family = _dataset_family(dataset_name)
    entity = _entity_name(feature_names, target)

    local_state_by_period = _build_periodic_local_state(seq_x)  # [3, 4]
    time_state = _time_and_state_features(ts, local_state_by_period)

    base = {
        "dataset_name": dataset_name,
        "dataset_family": family,
        "target": str(target),
        "entity": entity,
        "feature_dim": int(len(feature_names)),
        "timestamp": ts,
        "local_state_by_period": local_state_by_period.astype(np.float32),
        **time_state,
    }

    if family == "ett":
        domain_meta = _build_ett_meta(base)
    elif family == "electricity":
        domain_meta = _build_electricity_meta(base)
    elif family == "exchange_rate":
        domain_meta = _build_exchange_meta(base)
    elif family == "illness":
        domain_meta = _build_illness_meta(base)
    elif family == "traffic":
        domain_meta = _build_traffic_meta(base)
    elif family == "weather":
        domain_meta = _build_weather_meta(base)
    else:
        domain_meta = _build_generic_meta(base)

    meta_data = {**base, **domain_meta}

    # Unified ids expected by RetrievalTool.
    meta_data["dataset_id"] = _safe_hash(f"{family}:{dataset_name}", 2048)
    meta_data["sensor_type_id"] = _safe_hash(meta_data["sensor_type"], 256)
    meta_data["physical_location_id"] = _safe_hash(meta_data["physical_location"], 4096)

    # Ensure retrieval-required primitives exist and are int-compatible.
    meta_data["hour"] = int(meta_data["hour"])
    meta_data["day_of_week"] = int(meta_data["day_of_week"])
    meta_data["month"] = int(meta_data["month"])
    meta_data["is_holiday"] = int(meta_data["is_holiday"])
    meta_data["peak_status_id"] = int(meta_data["peak_status_id"])
    meta_data.pop("timestamp", None)

    meta_text = prompt_factory.build(meta_data)
    meta_text_by_period = prompt_factory.build_by_period(meta_data, local_state_by_period)
    return meta_data, meta_text, meta_text_by_period
