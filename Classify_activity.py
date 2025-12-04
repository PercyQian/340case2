# -*- coding: utf-8 -*-
"""
Improved activity recognition classifier using richer handcrafted features
and a Random Forest classifier, plus temporal smoothing of predictions.

This file is intended to replace the original logistic-regression demo in
Classify_activity.py while keeping the required predict_test() API.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

# Order of sensor axes in the input data:
#   'Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z'
sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']


def _extract_features(data):
    """
    Extract handcrafted features from raw IMU data.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, 60, 6)
        Raw time-series for each 1-minute window.

    Returns
    -------
    features : np.ndarray of shape (n_samples, n_features)
        Handcrafted feature vectors for each example.
    """
    # data: (N, T=60, D=6)
    N, T, D = data.shape
    # Basic statistics per axis
    mean_feat = np.mean(data, axis=1)              # (N, 6)
    std_feat = np.std(data, axis=1)                # (N, 6)
    min_feat = np.min(data, axis=1)                # (N, 6)
    max_feat = np.max(data, axis=1)                # (N, 6)
    median_feat = np.median(data, axis=1)          # (N, 6)

    # 25th and 75th percentiles -> interquartile range
    q25 = np.percentile(data, 25, axis=1)          # (N, 6)
    q75 = np.percentile(data, 75, axis=1)          # (N, 6)
    iqr_feat = q75 - q25                           # (N, 6)

    # Energy and mean absolute value
    energy_feat = np.mean(data ** 2, axis=1)       # (N, 6)
    abs_mean_feat = np.mean(np.abs(data), axis=1)  # (N, 6)

    # Linear trend (slope) over time for each axis
    t = np.arange(T)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = np.sum(t_centered ** 2)
    # Center the data per window and axis
    mean_expanded = mean_feat[:, np.newaxis, :]  # (N, 1, 6) -> broadcast
    data_centered = data - mean_expanded
    # Covariance with time: sum((t - mean_t) * (x - mean_x))
    num = np.sum(data_centered * t_centered[np.newaxis, :, np.newaxis],
                 axis=1)  # (N, 6)
    slope_feat = num / denom                      # (N, 6)

    # Concatenate all feature blocks: each is (N, 6)
    features = np.hstack(
        [
            mean_feat,
            std_feat,
            min_feat,
            max_feat,
            median_feat,
            iqr_feat,
            energy_feat,
            abs_mean_feat,
            slope_feat,
        ]
    )
    return features


def _extract_ecdf_features(data, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    N, T, D = data.shape
    q = np.percentile(data, percentiles, axis=1)
    q = np.moveaxis(q, 0, -1)
    quant_flat = q.reshape(N, -1)
    p_index = {p: i for i, p in enumerate(percentiles)}
    q1 = q[..., p_index[1]]
    q5 = q[..., p_index[5]]
    q10 = q[..., p_index[10]]
    q25 = q[..., p_index[25]]
    q50 = q[..., p_index[50]]
    q75 = q[..., p_index[75]]
    q90 = q[..., p_index[90]]
    q95 = q[..., p_index[95]]
    q99 = q[..., p_index[99]]
    d1 = (q95 - q5).reshape(N, -1)
    d2 = (q75 - q25).reshape(N, -1)
    d3 = (q50 - q25).reshape(N, -1)
    d4 = (q75 - q50).reshape(N, -1)
    d5 = (q99 - q1).reshape(N, -1)
    d6 = (q90 - q10).reshape(N, -1)
    features = np.hstack([quant_flat, d1, d2, d3, d4, d5, d6])
    return features


def _extract_magnitude_features(data, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    N, T, D = data.shape
    acc = data[:, :, :3]
    gyr = data[:, :, 3:]
    acc_mag = np.linalg.norm(acc, axis=2)
    gyr_mag = np.linalg.norm(gyr, axis=2)
    def feats_from_series(series):
        mean = np.mean(series, axis=1, keepdims=True)
        std = np.std(series, axis=1, keepdims=True)
        median = np.median(series, axis=1, keepdims=True)
        q25 = np.percentile(series, 25, axis=1).reshape(N, 1)
        q75 = np.percentile(series, 75, axis=1).reshape(N, 1)
        iqr = q75 - q25
        energy = np.mean(series ** 2, axis=1, keepdims=True)
        abs_mean = np.mean(np.abs(series), axis=1, keepdims=True)
        t = np.arange(T)
        t_centered = t - t.mean()
        series_centered = series - mean
        slope = (np.sum(series_centered * t_centered[np.newaxis, :], axis=1) / np.sum(t_centered ** 2)).reshape(N, 1)
        q = np.percentile(series, percentiles, axis=1)
        q = np.moveaxis(q, 0, -1)
        p_index = {p: i for i, p in enumerate(percentiles)}
        q1s = q[..., p_index[1]].reshape(N, 1)
        q5s = q[..., p_index[5]].reshape(N, 1)
        q10s = q[..., p_index[10]].reshape(N, 1)
        q25s = q[..., p_index[25]].reshape(N, 1)
        q50s = q[..., p_index[50]].reshape(N, 1)
        q75s = q[..., p_index[75]].reshape(N, 1)
        q90s = q[..., p_index[90]].reshape(N, 1)
        q95s = q[..., p_index[95]].reshape(N, 1)
        q99s = q[..., p_index[99]].reshape(N, 1)
        d1 = q95s - q5s
        d2 = q75s - q25s
        d3 = q50s - q25s
        d4 = q75s - q50s
        d5 = q99s - q1s
        d6 = q90s - q10s
        quant_flat = q.reshape(N, -1)
        return np.hstack([mean, std, median, iqr, energy, abs_mean, slope, quant_flat, d1, d2, d3, d4, d5, d6])
    acc_feats = feats_from_series(acc_mag)
    gyr_feats = feats_from_series(gyr_mag)
    return np.hstack([acc_feats, gyr_feats])


def _pairwise_corr_features(data):
    N, T, D = data.shape
    acc = data[:, :, :3]
    gyr = data[:, :, 3:]
    def corr_pairs(block):
        x = block[:, :, 0]
        y = block[:, :, 1]
        z = block[:, :, 2]
        def corr(a, b):
            ma = a.mean(axis=1, keepdims=True)
            mb = b.mean(axis=1, keepdims=True)
            ac = a - ma
            bc = b - mb
            cov = np.mean(ac * bc, axis=1)
            sa = a.std(axis=1) + 1e-8
            sb = b.std(axis=1) + 1e-8
            return (cov / (sa * sb)).reshape(N, 1)
        return np.hstack([corr(x, y), corr(x, z), corr(y, z)])
    acc_corr = corr_pairs(acc)
    gyr_corr = corr_pairs(gyr)
    return np.hstack([acc_corr, gyr_corr])


def _extract_rich_features(data):
    base = _extract_features(data)
    ecdf = _extract_ecdf_features(data)
    mag = _extract_magnitude_features(data)
    corr = _pairwise_corr_features(data)
    return np.hstack([base, ecdf, mag, corr])




def _smooth_predictions(preds, window_size=5):
    """
    Apply a simple temporal smoothing (majority filter) over predictions.

    Parameters
    ----------
    preds : np.ndarray of shape (n_samples,)
        Discrete label predictions.
    window_size : int, optional
        Size of the smoothing window (must be odd). Default is 5.

    Returns
    -------
    smoothed : np.ndarray of shape (n_samples,)
        Smoothed label sequence.
    """
    n = preds.shape[0]
    if window_size <= 1 or n == 0:
        return preds

    half = window_size // 2
    smoothed = preds.copy()

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = preds[start:end]
        vals, counts = np.unique(window, return_counts=True)
        max_count = counts.max()
        candidates = vals[counts == max_count]

        if candidates.size == 1:
            smoothed[i] = candidates[0]
        else:
            # Break ties by preferring the original prediction at position i,
            # otherwise fall back to the smallest label among candidates.
            if preds[i] in candidates:
                smoothed[i] = preds[i]
            else:
                smoothed[i] = candidates.min()

    return smoothed


def _smooth_proba(proba, classes, window_size=5):
    n = proba.shape[0]
    if window_size <= 1 or n == 0:
        return classes[np.argmax(proba, axis=1)]
    half = window_size // 2
    out = np.empty(n, dtype=classes.dtype)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        avg = proba[start:end].mean(axis=0)
        out[i] = classes[np.argmax(avg)]
    return out


def _estimate_hmm(train_labels, classes, alpha=1.0):
    K = classes.shape[0]
    idx = {c: i for i, c in enumerate(classes)}
    pi = np.full(K, alpha)
    for c in train_labels:
        pi[idx[c]] += 1.0
    pi = pi / pi.sum()
    trans = np.full((K, K), alpha)
    for i in range(len(train_labels) - 1):
        a = idx[train_labels[i]]
        b = idx[train_labels[i + 1]]
        trans[a, b] += 1.0
    trans = trans / trans.sum(axis=1, keepdims=True)
    return pi, trans


def _viterbi_decode(proba, pi, trans, classes):
    n, K = proba.shape
    log_pi = np.log(pi + 1e-12)
    log_trans = np.log(trans + 1e-12)
    log_em = np.log(proba + 1e-12)
    dp = np.empty((n, K))
    back = np.zeros((n, K), dtype=int)
    dp[0] = log_pi + log_em[0]
    for t in range(1, n):
        prev = dp[t - 1][:, None] + log_trans
        back[t] = np.argmax(prev, axis=0)
        dp[t] = log_em[t] + np.max(prev, axis=0)
    path = np.empty(n, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return classes[path]


def predict_test(train_data, train_labels, test_data):
    """
    Train a classifier on train_data / train_labels and predict labels for
    test_data.

    This implementation:
      1. Extracts a richer set of handcrafted features from each 1-minute
         window for all 6 sensor axes.
      2. Standardizes the feature space using only the training set.
      3. Trains a Random Forest classifier with class balancing.
      4. Applies a simple temporal smoothing over the sequence of
         predicted labels to respect activity continuity over time.

    Parameters
    ----------
    train_data : np.ndarray of shape (n_train, 60, 6)
        Training IMU data.
    train_labels : np.ndarray of shape (n_train,)
        Integer activity labels in {1, 2, 3, 4}.
    test_data : np.ndarray of shape (n_test, 60, 6)
        Test IMU data.

    Returns
    -------
    test_outputs : np.ndarray of shape (n_test,)
        Predicted activity labels for the test set.
    """
    # 1) Feature extraction
    train_features = _extract_rich_features(train_data)
    test_features = _extract_rich_features(test_data)

    # 2) Standardize features (fit on training data only)
    scaler = StandardScaler()
    train_features_std = scaler.fit_transform(train_features)
    test_features_std = scaler.transform(test_features)

    # 3) Train a robust, non-linear classifier
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(train_features_std, train_labels)
    proba = rf.predict_proba(test_features_std)
    pi, trans = _estimate_hmm(train_labels, rf.classes_, alpha=2.0)
    lam = 0.3
    trans = (1 - lam) * trans + lam * np.eye(trans.shape[0])
    path = _viterbi_decode(proba, pi, trans, rf.classes_)
    test_outputs = _smooth_predictions(path, window_size=3)

    return test_outputs


if __name__ == "__main__":
    # Optional quick sanity-check using a simple train/test split on a single
    # file (e.g., *_train_1.csv). This block will NOT run when imported by
    # the evaluation scripts.
    import matplotlib.pyplot as plt

    # Example debug usage (requires the CSV files to be in the working dir):
    try:
        labels = np.loadtxt("labels_train_1.csv", dtype=int)
        data_slice_0 = np.loadtxt(sensor_names[0] + "_train_1.csv", delimiter=",")
        data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1], len(sensor_names)))
        data[:, :, 0] = data_slice_0
        del data_slice_0
        for sensor_index in range(1, len(sensor_names)):
            data[:, :, sensor_index] = np.loadtxt(
                sensor_names[sensor_index] + "_train_1.csv", delimiter=","
            )

        # Simple extrapolation-style split: first 70% train, last 30% test
        n_total = labels.size
        split_idx = int(0.7 * n_total)
        train_data = data[:split_idx, :, :]
        train_labels = labels[:split_idx]
        test_data = data[split_idx:, :, :]
        test_labels = labels[split_idx:]

        test_outputs = predict_test(train_data, train_labels, test_data)

        from sklearn.metrics import f1_score

        micro_f1 = f1_score(test_labels, test_outputs, average="micro")
        macro_f1 = f1_score(test_labels, test_outputs, average="macro")
        print(f"Micro-averaged F1 score: {micro_f1:.4f}")
        print(f"Macro-averaged F1 score: {macro_f1:.4f}")

        train_features2 = _extract_ecdf_features(train_data)
        test_features2 = _extract_ecdf_features(test_data)
        scaler2 = StandardScaler()
        train_std2 = scaler2.fit_transform(train_features2)
        test_std2 = scaler2.transform(test_features2)
        rf2 = RandomForestClassifier(
            n_estimators=500,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        rf2.fit(train_std2, train_labels)
        proba2 = rf2.predict_proba(test_std2)
        proba_smooth = _smooth_proba(proba2, rf2.classes_, window_size=5)
        micro_f1_proba = f1_score(test_labels, proba_smooth, average="micro")
        macro_f1_proba = f1_score(test_labels, proba_smooth, average="macro")
        print(f"Micro F1 (proba smoothing): {micro_f1_proba:.4f}")
        print(f"Macro F1 (proba smoothing): {macro_f1_proba:.4f}")

        # Plot predictions vs. true labels
        n_test = test_labels.size
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(n_test), test_labels, "b.")
        plt.xlabel("Time window")
        plt.ylabel("Target")
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(n_test), test_outputs, "r.")
        plt.xlabel("Time window")
        plt.ylabel("Predicted target")
        plt.tight_layout()
        plt.show()
    except OSError:
        # CSV files not found; skip the debug run.
        print("CSV files not found; skipping __main__ debug run.")
