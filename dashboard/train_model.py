"""Rebuild the dashboard model from the original Kaggle CSV."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from detector import SENSORS, features, read_sensor_csv, score, sensor_window_means

HERE = Path(__file__).resolve().parent


def train(source):
    frame = read_sensor_csv(source)
    split = int(len(frame) * 0.70) // 15 * 15
    train_frame = frame.iloc[:split]
    normal = train_frame.loc[train_frame["machine_status"].eq("NORMAL"), SENSORS]
    quartiles = normal.quantile([0.25, 0.75])
    spread = quartiles.iloc[1] - quartiles.iloc[0]
    spread = spread.mask(spread.eq(0), np.inf)
    lower, upper = quartiles.iloc[0] - 3 * spread, quartiles.iloc[1] + 3 * spread
    medians = normal.median()
    windows, values, _ = features(train_frame, lower, upper, medians)
    normal_window = train_frame["machine_status"].eq("NORMAL").resample("15min").min()
    healthy = normal_window.rolling(4).sum().eq(4).reindex(windows.index).to_numpy()
    normal_values = values[healthy]
    calibration_start = int(len(normal_values) * 0.8)
    fit_values, calibration_values = np.split(normal_values, [calibration_start])
    scaler = StandardScaler().fit(fit_values)
    pca = PCA(n_components=10, svd_solver="full").fit(scaler.transform(fit_values))
    model = {
        "lower": lower.to_numpy(), "upper": upper.to_numpy(), "medians": medians.to_numpy(),
        "center": scaler.mean_, "scale": scaler.scale_, "pca_mean": pca.mean_,
        "components": pca.components_,
    }
    model["threshold"] = np.quantile(score(calibration_values, model), 0.95)
    np.savez_compressed(HERE / "pump_model.npz", **model)
    observed = sensor_window_means(train_frame).reindex(windows.index)
    reference = observed.loc[healthy].iloc[calibration_start:].quantile([0.01, 0.99])
    np.savez_compressed(HERE / "sensor_reference.npz",
                        lower=reference.iloc[0].to_numpy(),
                        upper=reference.iloc[1].to_numpy())

    held_out = frame.iloc[split:]
    windows, values, missing = features(held_out, lower, upper, medians)
    flagged = (score(values, model) > model["threshold"]) & missing.le(0.2)
    print(f"Chronological holdout: {len(windows):,} scored windows, {flagged.sum():,} flagged.")

    print(f"Trained on {len(fit_values):,} windows, calibrated on {len(calibration_values):,}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    train(parser.parse_args().csv)
