"""Shared features and inference for the pump PCA detector."""

import numpy as np
import pandas as pd

SENSORS = [f"sensor_{i:02d}" for i in range(52) if i not in (15, 50)]
WARMUP = 3


def read_sensor_csv(source):
    frame = pd.read_csv(source, nrows=250_001)
    if len(frame) > 250_000:
        raise ValueError("Use a CSV with at most 250,000 rows.")
    missing = [column for column in ["timestamp", *SENSORS] if column not in frame]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    timestamps = pd.to_datetime(frame.pop("timestamp"), format="mixed", errors="coerce")
    if timestamps.dtype.kind != "M" or isinstance(timestamps.dtype, pd.DatetimeTZDtype) or timestamps.isna().any():
        raise ValueError("Use valid timestamps without timezone offsets.")
    if timestamps.duplicated().any() or not timestamps.eq(timestamps.dt.floor("min")).all():
        raise ValueError("Timestamps must be unique and aligned to whole minutes.")
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    frame = frame.sort_index()
    if len(frame) < 60 or not (frame.index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=1)).all():
        raise ValueError("Provide at least 60 consecutive one-minute readings without timestamp gaps.")
    return frame


def features(frame, lower, upper, medians):
    readings = frame[SENSORS].apply(pd.to_numeric, errors="coerce")
    readings = readings.where(readings.ge(lower) & readings.le(upper))
    quarter_hours = readings.resample("15min").mean()
    full = readings.resample("15min").size().eq(15)
    quarter_hours = quarter_hours.loc[full]
    if len(quarter_hours) <= WARMUP:
        raise ValueError("Provide at least four complete clock-aligned 15-minute windows.")
    if quarter_hours.notna().sum().sum() == 0:
        raise ValueError("No usable sensor values remain after data validation.")
    missing_fraction = quarter_hours.isna().mean(axis=1)
    filled = quarter_hours.fillna(pd.Series(medians, index=SENSORS))
    rolling = filled.rolling(4)
    values = np.column_stack([filled, rolling.mean(), rolling.std(ddof=0)])[WARMUP:]
    return quarter_hours.iloc[WARMUP:], values, missing_fraction.rolling(4).max().iloc[WARMUP:]


def score(values, model):
    scaled = (values - model["center"]) / model["scale"] - model["pca_mean"]
    components = model["components"]
    reconstructed = (scaled @ components.T) @ components
    return np.mean((scaled - reconstructed) ** 2, axis=1)


def sample_readings(model):
    """Generate a reproducible 24-hour demo with injected sensor deviations."""
    periods = 96
    rng = np.random.default_rng(42)
    signals = np.tile(model["center"][:len(SENSORS)], (periods, 1))
    signals += rng.normal(0, 0.04, signals.shape) * model["scale"][:len(SENSORS)]
    for first, last in [(28, 35), (66, 76)]:
        for sensor in (0, 4, 6, 7, 8, 9, 10, 11):
            baseline = model["center"][sensor]
            boundary = model["upper"][sensor]
            if np.isfinite(boundary) and boundary > baseline:
                signals[first:last, sensor] = baseline + 0.9 * (boundary - baseline)
    index = pd.date_range("2025-01-01", periods=periods * 15, freq="min", name="timestamp")
    return pd.DataFrame(np.repeat(signals, 15, axis=0), index=index, columns=SENSORS)


def sensor_window_means(frame):
    """Use observed sensor values before the PCA-specific filtering and filling."""
    readings = frame[SENSORS].apply(pd.to_numeric, errors="coerce")
    readings = readings.where(np.isfinite(readings))
    windows = readings.resample("15min")
    complete = windows.size().eq(15)
    return windows.mean().where(windows.count().ge(12)).loc[complete]
