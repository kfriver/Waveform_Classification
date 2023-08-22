# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 15:17
# @Author  : longfei Kang
# @File    : train.py.py

import pandas as pd
from joblib import dump
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 读取到data的数据框csv文件
data = pd.read_csv('labeled_files.csv')

# 将类别变量编码为数值变量
data = pd.get_dummies(data)

# 分离特征和标签
X = data.drop('label', axis=1)
y = data['label']

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 查看特征数量
n_features = X.shape[1]
print(f'Number of features: {n_features}')

# 定义要使用的预测模型
estimator = SVC(kernel="linear")

# 定义 RFE 选择器并指定要选择的特征数量
selector = RFE(estimator, n_features_to_select=11)

# 使用 RFE 选择器来拟合数据并选择特征
X_new = selector.fit_transform(X, y)

# 另一种选择器，可以对比测试
# from sklearn.feature_selection import SelectKBest, f_classif
#
# # 使用SelectKBest选择与目标变量相关性最高的K个特征
# selector = SelectKBest(f_classif, k=10)
# X_new = selector.fit_transform(X, y)

# 调整决策树模型的参数
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=0.2)

# 使用训练集来训练模型
clf.fit(X_train, y_train)

# 在验证集上评估模型性能
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')
print(y_pred)
print('Accuracy:', accuracy)
print('Recall:', recall)
print('F1 Score:', f1)

# 保存模型文件
dump(clf, 'classifier.joblib')
