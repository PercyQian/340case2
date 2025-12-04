# HAR Case Study 计划文档

## 目标与数据
- 任务：Human Activity Recognition（HAR），对 1 分钟窗口进行四分类：Resting、Walking、Running、Commuting。
- 输入：`(N, 60, 6)`，60 秒×6 轴（Acc_x, Acc_y, Acc_z, Gyr_x, Gyr_y, Gyr_z）。
- 输出：测试集标签预测，接口：`predict_test(train_data, train_labels, test_data)`。

## 约束与评估
- 不随机打乱；不做标准 K 折；时间相关性强，打乱会泄漏。
- 验证方式：时间顺序 Holdout（如前 2/3 训练，后 1/3 测试）。
- 离线预测允许：可使用测试集特征做后处理（不可用测试标签）。
- 指标：micro-F1 与 macro-F1；需至少超过 Mean/Std+ElasticNet-LR 基线。

## 流程设计
- Pre-processing：基于 ECDF 的分位数与分位差特征（每轴）。
- Classification：`RandomForestClassifier`，`class_weight="balanced"`，多树稳定。
- Post-processing：时间平滑，默认多数投票；可选概率平滑；需要时可扩展 HMM。

## 特征工程（ECDF）
- 分位数网格：`p = [5, 25, 50, 75, 95]`；每轴提取分位值与差值。
- 特征项：`Q(p)` 值；差值 `Q95-Q5`, `Q75-Q25`, `Q50-Q25`, `Q75-Q50`。
- 维度：每轴 9 项，6 轴合计 54 维；可按表现扩展分位数。
- 仅在训练段拟合任何变换；避免测试信息泄漏。

## 分类算法（Random Forest）
- 训练：ECDF 特征→（可选标准化）→随机森林。
- 超参初始值：`n_estimators=500`, `max_depth=24`, `min_samples_leaf=2`, `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`。
- 输出：`predict` 得到离散标签；`predict_proba` 供概率平滑或 HMM 使用。

## 后处理（平滑）
- 多数投票：滑窗大小 5~7，边界截断；默认启用。
- 概率平滑：对 `predict_proba` 序列做均值滤波后取 `argmax`。
- HMM（可选）：训练段统计转移矩阵；发射用分类器概率；Viterbi 解码。

## 验证流程
- 时间顺序切分训练/测试（如 70/30）。
- 训练与特征拟合仅用训练段。
- 在测试段上：生成原始预测；应用多数投票与概率平滑并比较 F1。
- 代码入口：`c:\Users\24457\Desktop\340case2\Classify_activity.py:125`。

## 不该做的事
- 不打乱数据；不做标准 K 折。
- 不用测试标签进行任何训练、调参或平滑。
- 不在训练拟合中使用测试段统计量。
- 若已知序列边界，不跨边界平滑。

## 交付物与代码变更
- 新特征函数：`_extract_ecdf_features`。
- 可选概率平滑函数：`_smooth_proba`。
- `predict_test` 切换到 ECDF 特征；保留随机森林与多数投票平滑。
- 在 `__main__` 中增加多数投票与概率平滑 F1 对比。
