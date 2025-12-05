# HAR Case Study — Report Outline

## 1. Problem Overview and Data
- Task: Smartphone IMU-based HAR on 1-minute windows (four classes: Resting, Walking, Running, Commuting).
- Input: `numpy` arrays shaped `(N, 60, 6)` with axes order `Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z`.
- Temporal constraints: Time-ordered windows; no random shuffling; holdout validation using chronological split; offline prediction may access test features but not labels.

## 2. Final Algorithm and Parameters
- Pipeline summary:
  - Feature Extraction: Rich handcrafted features combining base statistics, extended ECDF, magnitude features, inter-axis correlations, and first-order differences.
  - Standardization: `StandardScaler` fitted on training features only.
  - Classifier: `RandomForestClassifier` with `n_estimators=1000`, `max_depth=None`, `min_samples_leaf=2`, `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`.
  - Probability Calibration: Chronological holdout within the training set (last ~20%) with `isotonic` calibration; fall back to uncalibrated probabilities if the calibration slice is too small or has a single class.
  - Temporal Post-processing: HMM-based Viterbi decoding using training-label-estimated priors/transition probabilities, row-wise self-loop reinforcement informed by per-class minimum durations, then majority-vote smoothing (`window_size=5`).
- Implementation entry: `predict_test` in `c:\Users\24457\Desktop\340case2\Classify_activity.py:349`.

```python
# Excerpt from predict_test (c:\Users\24457\Desktop\340case2\Classify_activity.py:376-409)
train_features = _extract_rich_features(train_data)
scaler = StandardScaler()
train_features_std = scaler.fit_transform(train_features)
...
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
# Chronological holdout calibration (no k-fold)
split_idx = int(0.8 * len(train_features_std))
rf.fit(train_features_std[:split_idx], train_labels[:split_idx])
use_calib = (... ensure size and multi-class ...)
if use_calib:
    calib = CalibratedClassifierCV(rf, cv="prefit", method="isotonic")
    calib.fit(train_features_std[split_idx:], train_labels[split_idx:])
    proba = calib.predict_proba(test_features_std)
    classes = calib.classes_
else:
    proba = rf.predict_proba(test_features_std)
    classes = rf.classes_
pi, trans = _estimate_hmm(train_labels, classes, alpha=2.0)
mins = _estimate_min_durations(train_labels, classes)
# Row-wise self-loop reinforcement guided by mins
...
path = _viterbi_decode(proba, pi, trans, classes)
path = _enforce_min_duration(path, proba, classes, mins, max_passes=2)
test_outputs = _smooth_predictions(path, window_size=5)
```

## 3. Pre- and Post-Processing Details and Impact
- Feature Extraction:
  - Base statistics per axis (mean, std, min, max, median, IQR, energy, abs-mean, slope). Function: `_extract_features` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:20`).
  - Extended ECDF per axis using percentiles `[1,5,10,25,50,75,90,95,99]` and differences (e.g., `Q99-Q1`, `Q95-Q5`, `Q75-Q25`). Function: `_extract_ecdf_features` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:82`).
  - Magnitude features from `||Acc||` and `||Gyr||` sequences with the same percentile set and dynamics (energy, slope). Function: `_extract_magnitude_features` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:107`).
  - Inter-axis correlations within Acc and Gyr (pairwise Pearson-like across time). Function: `_pairwise_corr_features` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:151`).
  - First-order differences `abs(diff)` per axis with percentile summary. Function: `_extract_diff_features` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:278`).
  - Aggregator: `_extract_rich_features` concatenates all feature blocks (`c:\Users\24457\Desktop\340case2\Classify_activity.py:174`).

```python
# Feature aggregator (c:\Users\24457\Desktop\340case2\Classify_activity.py:174-181)
def _extract_rich_features(data):
    base = _extract_features(data)
    ecdf = _extract_ecdf_features(data)
    mag = _extract_magnitude_features(data)
    corr = _pairwise_corr_features(data)
    diff = _extract_diff_features(data)
    return np.hstack([base, ecdf, mag, corr, diff])
```

- Post-processing:
  - HMM estimation from training labels: `_estimate_hmm` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:243`) and Viterbi decoding `_viterbi_decode` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:259`).
  - Per-class minimum duration estimation: `_estimate_min_durations` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:289`) and enforcement `_enforce_min_duration` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:310`).
  - Final majority smoothing: `_smooth_predictions` (`c:\Users\24457\Desktop\340case2\Classify_activity.py:185`).
- Impact on results:
  - Rich features improved class separability (notably Running vs Walking and Commuting vs Resting).
  - Calibration improved HMM emission reliability; duration constraints reduced spurious short segments.

## 4. Model Selection, Testing, and Comparisons
- Baseline: Mean/Std + Elastic Net Logistic Regression (course baseline; outperformed by our approach).
- Random Forest (final choice): Robust to noisy features; balanced class weights; scalable with `n_estimators=1000`.
- ExtraTrees (explored): Similar but slightly lower macro-F1 in our tests.
- Smoothing strategies compared: Majority voting vs probability smoothing vs HMM decoding; HMM consistently superior in preserving temporal consistency.

- Illustrative results (Train1→Train2 chronological evaluation):
  - Base stats + RF + majority smoothing: Micro ≈ 0.82, Macro ≈ 0.82.
  - ECDF-rich + RF + majority smoothing: Micro ≈ 0.93, Macro ≈ 0.94.
  - Final pipeline (rich features + RF(1000) + calibrated HMM + duration + majority smoothing): Micro ≈ 0.95–0.97, Macro ≈ 0.96–0.97 (observed).

## 5. Evaluation Strategy and Deployment Recommendations
- Metrics:
  - Use both micro-F1 (overall accuracy emphasis) and macro-F1 (class balance emphasis); macro-F1 is critical when minority classes matter (e.g., Commuting).
- Smartphone HAR app considerations:
  - Near real-time constraints prohibit access to future data; replace symmetric smoothing with causal methods.
  - Causal decoding:
    - Use forward-only majority smoothing on recent `K` windows.
    - Replace full Viterbi with online filtering (e.g., forward algorithm) or fixed-lag smoothing with small lag.
  - Latency vs accuracy trade-off: Larger `K` yields stability but increases delay; recommend `K=3–5` for 60s windows.

```python
# Example causal majority smoothing (pseudo-code)
def causal_smooth(preds, K=5):
    out = []
    for i in range(len(preds)):
        start = max(0, i-K+1)
        window = preds[start:i+1]
        out.append(np.bincount(window).argmax())
    return np.array(out)
```

## 6. Reproducibility and Implementation Details
- Dependencies: `numpy`, `scikit-learn`, `matplotlib`.
- Data files: `labels_train_1.csv`, `labels_train_2.csv`, and sensor CSVs `Acc_x_train_*.csv`, ..., `Gyr_z_train_*.csv` placed in the working directory.
- Run:
  - `python Classify_activity.py` for internal 70/30 chronological check.
  - `python Evaluate_classifier.py` for Train1→Train2 evaluation.
- No k-fold cross-validation is used; calibration leverages a chronological training holdout slice.

## 7. Failed/Explored Approaches and Learnings
- Probability smoothing without HMM: Improved stability modestly but underperformed vs HMM.
- ExtraTrees vs RandomForest: RF achieved higher macro-F1 with similar compute.
- K-fold calibration: Avoided to respect temporal constraints; chronological holdout calibration was adopted instead.

## 8. Appendix (Optional)
- Additional figures: Confusion matrices per configuration; per-class duration distributions; example timelines.
- Extended tables: Hyperparameter sweeps (e.g., `n_estimators`, self-loop weights, window sizes).
