# -*- coding: utf-8 -*-
"""
Improved activity recognition classifier using richer handcrafted features
and a Random Forest classifier, plus temporal smoothing of predictions.

This file is intended to replace the original logistic-regression demo in
Classify_activity.py while keeping the required predict_test() API.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    train_features = _extract_features(train_data)
    test_features = _extract_features(test_data)

    # 2) Standardize features (fit on training data only)
    scaler = StandardScaler()
    train_features_std = scaler.fit_transform(train_features)
    test_features_std = scaler.transform(test_features)

    # 3) Train a robust, non-linear classifier
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(train_features_std, train_labels)
    raw_preds = rf.predict(test_features_std)

    # 4) Temporal smoothing of predictions
    test_outputs = _smooth_predictions(raw_preds, window_size=5)

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
