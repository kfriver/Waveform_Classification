# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 15:18
# @Author  : longfei Kang
# @File    : cls_model.py

import os
import numpy as np
import scipy.stats
from joblib import load
import matplotlib.pyplot as plt

# 加载训练好的模型
clf = load('classifier.joblib')


# 定义函数来提取波形特征
def extract_features(waveform):
    mean = np.mean(waveform)
    std = np.std(waveform)
    max_value = np.max(waveform)
    min_value = np.min(waveform)
    median = np.median(waveform)
    kurtosis = scipy.stats.kurtosis(waveform)
    skew = scipy.stats.skew(waveform)
    rms = np.sqrt(np.mean(np.square(waveform)))
    crest_factor = np.max(np.abs(waveform)) / rms
    impulse_factor = np.max(np.abs(waveform)) / np.mean(np.abs(waveform))
    clearance_factor = np.max(np.abs(waveform)) / np.mean(np.square(waveform))
    return [mean, std, max_value, min_value, median, kurtosis, skew, rms, crest_factor, impulse_factor,
            clearance_factor]


# 定义函数来读取文件内容并提取波形特征
def read_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 检查文件内容是否为空
    if not content.strip():
        print(f'Deleting empty file: {file_path}')
        os.remove(file_path)
        return None, None

    # 检查文件内容是否包含非数值数据
    try:
        waveform = [float(x) for x in content.split()]
    except ValueError:
        print(f'File contains non-numeric data: {file_path}')
        return None, None

    features = extract_features(waveform)
    return waveform, features


# 读取未分类数据并提取特征
data_dir = './data_origin'  # 此处需要填写数据文件所在目录,可以就放在data_origin目录下
X = []
waveforms = []
file_names = []
for file_name in os.listdir(data_dir):
    if file_name.endswith('.ini'):
        file_path = os.path.join(data_dir, file_name)
        waveform, features = read_file(file_path)

        # 检查波形和特征是否为空
        if waveform is None or features is None:
            continue

        X.append(features)
        waveforms.append(waveform)
        file_names.append(file_name)

# 使用训练好的模型对未分类数据进行分类
y_pred = clf.predict(X)

# 根据预测结果将未分类数据保存到不同的文件夹中
for i in range(len(y_pred)):
    label = y_pred[i]
    file_name = file_names[i]
    waveform = waveforms[i]
    dst_dir = os.path.join('./classified_data', str(label))  # 分类好的文件会被保存在classified_data中，0是低噪声，1是高噪声
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, file_name)
    src_path = os.path.join(data_dir, file_name)
    os.rename(src_path, dst_path)

    # 保存波形图
    plt.plot(waveform)
    plt.savefig(os.path.join(dst_dir, f'{os.path.splitext(file_name)[0]}.png'))
    plt.close()
