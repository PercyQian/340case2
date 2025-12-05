# -*- coding: utf-8 -*-
"""
Script used for pre-submission evaluation of classifier accuracy on training
set 2

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from Classify_activity import predict_test, _extract_rich_features, sensor_names

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'
test_suffix = '_train_2.csv'

def load_sensor_data(sensor_names, suffix):
    data_slice_0 = np.loadtxt(sensor_names[0] + suffix, delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                     len(sensor_names)))
    data[:, :, 0] = data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            sensor_names[sensor_index] + suffix, delimiter=',')
    
    return data
    
# Load labels and sensor data into 3-D array
train_labels = np.loadtxt('labels' + train_suffix, dtype='int')
train_data = load_sensor_data(sensor_names, train_suffix)
test_labels = np.loadtxt('labels' + test_suffix, dtype='int')
test_data = load_sensor_data(sensor_names, test_suffix)

# Predict activities on test data
test_outputs = predict_test(train_data, train_labels, test_data)

# Compute micro and macro-averaged F1 scores
micro_f1 = f1_score(test_labels, test_outputs, average='micro')
macro_f1 = f1_score(test_labels, test_outputs, average='macro')
print(f'Micro-averaged F1 score: {micro_f1}')
print(f'Macro-averaged F1 score: {macro_f1}')

# Examine outputs compared to labels
n_test = test_labels.size
plt.subplot(2, 1, 1)
plt.plot(np.arange(n_test), test_labels, 'b.')
plt.xlabel('Time window')
plt.ylabel('Target')
plt.subplot(2, 1, 2)
plt.plot(np.arange(n_test), test_outputs, 'r.')
plt.xlabel('Time window')
plt.ylabel('Output (predicted target)')
plt.show()
    
def build_feature_names(sensor_names):
    names = []
    stats = ["mean","std","min","max","median","iqr","energy","abs_mean","slope"]
    for s in stats:
        for ax in sensor_names:
            names.append(f"base_{s}_{ax}")
    ecdf_ps = (1,5,10,25,50,75,90,95,99)
    for p in ecdf_ps:
        for ax in sensor_names:
            names.append(f"ecdf_p{p}_{ax}")
    diffs = [(95,5),(75,25),(50,25),(75,50),(99,1),(90,10)]
    for a,b in diffs:
        for ax in sensor_names:
            names.append(f"ecdf_d{a}_{b}_{ax}")
    mag_stats = ["mean","std","median","iqr","energy","abs_mean","slope"]
    for s in mag_stats:
        names.append(f"mag_acc_{s}")
    for p in ecdf_ps:
        names.append(f"mag_acc_p{p}")
    for a,b in diffs:
        names.append(f"mag_acc_d{a}_{b}")
    for s in mag_stats:
        names.append(f"mag_gyr_{s}")
    for p in ecdf_ps:
        names.append(f"mag_gyr_p{p}")
    for a,b in diffs:
        names.append(f"mag_gyr_d{a}_{b}")
    corr_pairs = ["xy","xz","yz"]
    for cp in corr_pairs:
        names.append(f"corr_acc_{cp}")
    for cp in corr_pairs:
        names.append(f"corr_gyr_{cp}")
    for ax in sensor_names:
        names.append(f"diff_mean_{ax}")
    for ax in sensor_names:
        names.append(f"diff_std_{ax}")
    for p in ecdf_ps:
        for ax in sensor_names:
            names.append(f"diff_p{p}_{ax}")
    return names

train_features_imp = _extract_rich_features(train_data)
scaler_imp = StandardScaler()
train_std_imp = scaler_imp.fit_transform(train_features_imp)
rf_imp = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf_imp.fit(train_std_imp, train_labels)
importances = rf_imp.feature_importances_
feature_names = build_feature_names(sensor_names)
group_sizes = [9*len(sensor_names), 9*len(sensor_names) + 6*len(sensor_names), 22*2, 6, 6 + 6 + 9*len(sensor_names)]
idx0 = 0
group_vals = []
for gs in group_sizes:
    group_vals.append(importances[idx0:idx0+gs].sum())
    idx0 += gs
group_labels = ["Base","ECDF","Magnitude","Correlation","Diff"]
order = np.argsort(group_vals)[::-1]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.bar([group_labels[i] for i in order],[group_vals[i] for i in order],color=["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f"])
plt.ylabel("Sum Importance")
plt.title("Feature Importance by Group")
topk = 20
order_feat = np.argsort(importances)[::-1][:topk]
plt.subplot(1,2,2)
plt.barh(range(topk), importances[order_feat][::-1], color="#4e79a7")
plt.yticks(range(topk), [feature_names[i] for i in order_feat][::-1])
plt.xlabel("Importance")
plt.title("Top-20 Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.show()
