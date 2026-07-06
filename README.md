# Anomaly Detection in Pump Sensor Data

**Course:** BFVM25DATASC6 — Data Science  
**Author:** Ali Akhlaghi
**Assignment:** Portfolio Assignment — Anomaly Detection

---

## Overview

This project applies unsupervised anomaly detection to multivariate time-series data collected from industrial pump sensors. The goal is to detect early warning signs of machine failure **before** a breakdown occurs, enabling predictive maintenance.

The dataset contains 220,320 minute-level readings from 52 sensors, with machine status labels (`NORMAL`, `RECOVERING`, `BROKEN`). Only 7 `BROKEN` events exist across 5 months of data, making this a highly imbalanced problem that motivates an unsupervised approach.

---

## Project Structure

```
anomaly_detection/
├── Final_Assignment_Ali.ipynb   # Main notebook (all analysis and results)
├── config.yaml                  # Data path configuration
├── README.md                    # This file
└── report.md                    # Summary report
```

**Data** (not included — download separately):
```
study_cases/data/sensor.csv
```
Download from: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

---

## Hypotheses

| # | Hypothesis |
|---|-----------|
| H1 | Sensor behavior changes before machine failure events |
| H2 | Rolling mean and rolling std improve detection compared to raw values |
| H3 | A multivariate model capturing sensor correlations performs better |
| H4 | Proper preprocessing significantly improves detection performance |

---

## Methods

### Data Preprocessing (Task 2)
- Drop `sensor_15` (100% missing) and `sensor_50` (35% missing)
- **IQR-based sensor-error removal** — replace values beyond ±3×IQR with NaN and interpolate
- **15-minute resampling** — reduces noise and dataset size
- **Rolling features** — 60-minute rolling mean and std per sensor (150 features total)

### Models (Task 1 — Baseline)
Four algorithms from the course Study Case notebook, evaluated on both original and improved data:

| Algorithm | Type |
|-----------|------|
| Isolation Forest | Tree-based |
| Local Outlier Factor (LOF) | Density-based |
| One-Class SVM | Kernel / boundary |
| Robust Covariance | Statistical / Gaussian |

### New Algorithm (Task 3)
**PCA Reconstruction-Error Anomaly Detector** — trained on normal data only; anomalies generate high reconstruction error. Hyperparameter tuned over `n_components ∈ {5, 10, 15, 20, 30, 40, 50}`.

---

## Requirements

```
Python 3.12
scikit-learn
pandas
numpy
matplotlib
seaborn
scipy
pyyaml
jupyter
```

Install via:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy pyyaml jupyter
```

---

## How to Run

1. Download `sensor.csv` from Kaggle and place it at `study_cases/data/sensor.csv`
2. Open `Final_Assignment_Ali.ipynb` in Jupyter
3. Select the correct Python kernel (with sklearn installed)
4. Run **Kernel → Restart & Run All**

---

## References

1. Liu et al. (2012). *Isolation-based anomaly detection*. ACM TKDD.
2. Breunig et al. (2000). *LOF: Identifying density-based local outliers*. ACM SIGMOD.
3. Shyu et al. (2003). *A novel anomaly detection scheme based on PCA*. IEEE Workshop.
4. Darban et al. (2024). *Deep Learning for Time Series Anomaly Detection: A Survey*. ACM Comput. Surv. https://doi.org/10.1145/3691338
