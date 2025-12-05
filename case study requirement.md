Project Overview

Task: Human Activity Recognition (HAR) using smartphone Inertial Measurement Unit (IMU) data.


Goal: Classify 1-minute time windows into one of 4 physical tasks.

Classes (Labels):

Resting (standing/sitting).

Walking.

Running.

Commuting (car).

Data Specifications

Input Shape: A 3D numpy array: (num_examples, 60, 6).



Dimension 1: Number of 1-minute time windows (examples).


Dimension 2: 60 seconds (1 data point per second).


Dimension 3: 6 Sensor Axes in this order: 'Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z'.


Temporal Nature: Data is time-ordered. Rows are often consecutive, but gaps may exist (e.g., stopping one day and starting the next).

Critical Constraints (Do Not Ignore)
No Random Shuffling: You must not shuffle data or use standard K-fold cross-validation. The data implies time dependence; shuffling causes data leakage.


Validation Strategy: Use Holdout validation (interpolation or extrapolation). E.g., Train on the first 2/3, test on the last 1/3 .


Offline Prediction: You are allowed to access the test set features (but not labels) during prediction, allowing for semi-supervised approaches or post-processing.

Evaluation Metrics
Scoring: Accuracy is based on two metrics on a held-out test set:


Micro-averaged F1 score.


Macro-averaged F1 score.

Baselines:


Non-competitive (Must Beat): Feature extraction (Mean/Std) + Elastic Net Logistic Regression .


Competitive: A higher accuracy threshold established by the course staff.

Code Requirements

File Name: Classify_activity.py.

Required Function Signature:

Python

def predict_test(train_data, train_labels, test_data):
train_data: (N, 60, 6) numpy array.

train_labels: (N,) vector.


test_data: (M, 60, 6) numpy array.


Output: The function must return predicted labels for the test_data.

Recommended Approaches

Feature Extraction (Shallow): Calculate statistics (Mean, Std, Max, Min) for each axis over the 60-second window, then use Logistic Regression or Random Forest.




Deep Learning: Use CNNs (treating sensors as channels) or RNNs/LSTMs on the raw sequence data.

Post-Processing: Use temporal smoothing (e.g., if neighbors are "Walking", the middle is likely "Walking") to correct errors.