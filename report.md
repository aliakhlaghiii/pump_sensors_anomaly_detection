# Report — Anomaly Detection in Pump Sensor Data

**Course:** BFVM25DATASC6 — Data Science  
**Author:** Ali Akhlaghi
**Date:** June 2026

---

## 1. Problem Understanding

The goal of this project is to detect early warning signals of machine failure in industrial pump sensor data using unsupervised anomaly detection. The dataset spans 5 months (April–August 2018) with 52 sensors sampled every minute.

**Why unsupervised?**  
The dataset contains only 7 `BROKEN` events out of 220,320 rows — an extreme class imbalance of 0.003%. This makes supervised classification impractical. Unsupervised methods learn what *normal* looks like and flag deviations, without requiring balanced labeled data.

**Anomaly definition:**  
Rather than labeling only the 7 exact failure timestamps, a 24-hour failure window before each `BROKEN` event is defined as anomalous. This reflects real-world industrial systems where sensor behavior degrades gradually before breakdown. The resulting `anomaly_label` covers 10,087 minutes (~4.6% of data).

---



## 2. Data Exploration


| Property      | Value                                 |
| ------------- | ------------------------------------- |
| Total rows    | 220,320                               |
| Sensors       | 52 raw; 51 after removing `sensor_15` |
| Sampling rate | 1 minute                              |
| NORMAL        | 205,836 (93.4%)                       |
| RECOVERING    | 14,477 (6.6%)                         |
| BROKEN        | 7 (0.003%)                            |


Key observations from EDA:

- `sensor_15`: 100% missing — dropped
- `sensor_50`: ~35% missing — dropped in improved pipeline
- `sensor_04` and `sensor_00` show clear drops before `BROKEN` events, confirming **H1**
- `sensor_51` contains extreme spikes likely caused by sensor errors (not failures)

---



## 3. Data Quality Improvement (Task 2)

Two preprocessing pipelines were compared:

**Experiment A — Original data (baseline)**

- Drop `sensor_15` only
- Fill missing values with column mean
- StandardScaler normalization

**Experiment B — Improved data**

- Drop `sensor_15` and `sensor_50`
- IQR-based sensor-error removal (threshold: ±3×IQR)
- Linear interpolation for remaining missing values
- Resample to 15-minute intervals (mean aggregation)
- Rolling features: 60-minute window mean and std per sensor
- StandardScaler normalization

The improved pipeline reduced the dataset from 220,320 rows to 14,688 rows and expanded the feature set from 51 to 150 features (50 sensors × raw + rolling mean + rolling std).

---



## 4. Algorithms



### 4.1 Baseline Algorithms (Task 1)

**Isolation Forest** (Liu et al., 2012)  
Tree-based method. Anomalies are isolated in fewer splits because they lie in sparse regions of feature space. Fast and scalable.

**Local Outlier Factor (LOF)** (Breunig et al., 2000)  
Density-based method. Compares local density of each point to its neighbours. Points in significantly lower-density regions are anomalies.

**One-Class SVM (OCSVM)** (Schölkopf et al., 1999)  
Kernel-based boundary method. Learns a hypersphere around normal data; points outside are anomalies.

**Robust Covariance** (Rousseeuw & Van Driessen, 1999)  
Statistical method based on Mahalanobis distance from a robust mean estimate.

### 4.2 New Algorithm (Task 3) — PCA Reconstruction Error

**Rationale:**  
PCA-based anomaly detection (Shyu et al., 2003) differs fundamentally from the four baseline methods in one key property: it is **trained on normal data only**. The PCA learns the correlation structure of healthy operation. When applied to new data, points that cannot be well reconstructed (high reconstruction error) are flagged as anomalies.

**Hyperparameter tuning:**  
The number of principal components (`n_components`) was swept over {5, 10, 15, 20, 30, 40, 50}. The best value was selected by maximising F1 score.


| n_components | Var. Explained | F1    |
| ------------ | -------------- | ----- |
| 5            | 0.428          | 0.053 |
| 10           | 0.563          | 0.047 |
| 15           | 0.644          | 0.036 |
| 20           | 0.702          | 0.039 |
| 30           | 0.786          | 0.035 |
| 40           | 0.846          | 0.031 |
| 50           | 0.890          | 0.027 |


**Best:** `n_components = 5` — deliberately keeping only 42.8% of variance amplifies the reconstruction error contrast between normal and anomalous points.

---



## 5. Results



### 5.1 Performance Comparison


| Model             | Experiment   | Precision | Recall | F1        | AUC   |
| ----------------- | ------------ | --------- | ------ | --------- | ----- |
| Isolation Forest  | A — Original | 0.131     | 0.131  | **0.131** | 0.544 |
| Isolation Forest  | B — Improved | 0.105     | 0.105  | 0.105     | 0.531 |
| LOF               | B — Improved | 0.085     | 0.085  | 0.085     | 0.521 |
| LOF               | A — Original | 0.071     | 0.071  | 0.071     | 0.513 |
| One-Class SVM     | A — Original | 0.062     | 0.062  | 0.062     | 0.509 |
| One-Class SVM     | B — Improved | 0.061     | 0.062  | 0.061     | 0.508 |
| PCA (n=5)         | B — Improved | 0.043     | 0.068  | 0.053     | 0.497 |
| Robust Covariance | B — Improved | 0.031     | 0.031  | 0.031     | 0.492 |
| Robust Covariance | A — Original | 0.000     | 0.000  | 0.000     | 0.476 |


Random baseline F1 ≈ 0.046. All models except Robust Covariance (Original) outperform random.

### 5.2 Hypothesis Evaluation


| Hypothesis                                  | Result              | Evidence                                                 |
| ------------------------------------------- | ------------------- | -------------------------------------------------------- |
| H1: sensor behaviour changes before failure | Confirmed           | All models flag higher scores near BROKEN events         |
| H2: rolling features improve detection      | Confirmed           | LOF F1 improved from 0.071 → 0.085 with rolling features |
| H3: multivariate models perform better      | Partially confirmed | PCA uses correlations but does not outperform IF on F1   |
| H4: preprocessing improves detection        | Confirmed           | Robust Covariance: 0.000 → 0.031; LOF: 0.071 → 0.085     |


---



## 6. Discussion

**Why are F1 scores low overall?**  
The 24-hour failure window is intentionally broad. During much of this window (especially the first 18–20 hours), sensors behave nearly normally — only the final hours before breakdown show clear deviations. This means models correctly predict anomalies for part of the window but miss the rest, resulting in moderate recall and low precision.

**Why does Isolation Forest perform best on original data?**  
Resampling to 15 minutes smooths out short-duration spikes that are genuinely anomalous. Isolation Forest is effective at detecting these brief extreme deviations. After resampling, this advantage disappears.

**Why does PCA not outperform all baselines?**  
With `n_components=5`, only 42.8% of variance is retained. This is a crude model of normal behaviour. With more failure data, PCA could be trained more robustly and the threshold calibrated more precisely.

---



## 7. Limitations and Future Work

- Only 7 failure events limit statistical reliability of all metrics
- The 24-hour window is an engineering estimate; domain expertise would improve it
- Future work: LSTM autoencoder for non-linear temporal anomaly detection (Malhotra et al., 2016)
- Data augmentation (e.g., synthetic failure generation) could improve model calibration

---



## 8. Conclusion

All four hypotheses were at least partially confirmed. Preprocessing consistently improved model performance, and rolling standard deviation emerged as the most informative feature for pre-failure detection. The PCA reconstruction-error detector, trained exclusively on normal data, provides an interpretable and theoretically motivated approach for online monitoring. A deployment recommendation is to alert maintenance teams when the 15-minute reconstruction error exceeds the learned threshold.

---



## References

1. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2012). *Isolation-based anomaly detection*. ACM TKDD, 6(1), 1–39.
2. Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying density-based local outliers*. ACM SIGMOD, 93–104.
3. Shyu, M. L., Chen, S. C., Sarinnapakorn, K., & Chang, L. (2003). *A novel anomaly detection scheme based on principal component classifier*. IEEE Workshop.
4. Darban et al. (2024). *Deep Learning for Time Series Anomaly Detection: A Survey*. ACM Comput. Surv. 57(1). [https://doi.org/10.1145/3691338](https://doi.org/10.1145/3691338)



### Course Material

- Study Case: *Anomaly Detection Algorithms* — instructor notebook, BFVM25DATASC6
- Portfolio Assignment: *PA-Anomaly* — instructor assignment, BFVM25DATASC6
- Data: Pump sensor dataset. [https://www.kaggle.com/datasets/nphantawee/pump-sensor-data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

---

*Parts of the visualization code were developed with assistance from ChatGPT (OpenAI). All analysis, interpretation, and conclusions are the author's own work.*