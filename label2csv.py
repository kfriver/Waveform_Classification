# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 15:22
# @Author  : longfei Kang
# @File    : label2csv.py

import os
import pandas as pd
import numpy as np
import scipy.stats

data = pd.DataFrame(
    columns=['mean', 'std', 'max', 'min', 'median', 'kurtosis', 'skew', 'rms', 'crest_factor', 'impulse_factor',
             'clearance_factor', 'label'])
# 将已经分类好的 待训练数据放在data目录下的0，1 文件夹中。0是低噪声波形，1是高噪声波形
for i in range(2):
    for file_name in os.listdir(os.path.join('./data', str(i))):
        if file_name.endswith('.ini'):
            # 读取文件内容
            with open(os.path.join('./data', str(i), file_name), 'r') as f:
                content = f.read()

            # 检查文件内容是否为空
            if not content.strip():
                print(f'Empty file: {os.path.join("./data", str(i), file_name)}')
                continue

            # 检查文件内容是否包含非数值数据
            try:
                waveform = [float(x) for x in content.split()]
            except ValueError:
                print(f'File contains non-numeric data: {os.path.join("./data", str(i), file_name)}')
                continue

            # 计算波形的统计特征
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

            # 将特征和标签添加到数据框中
            label = i
            data = pd.concat([data, pd.DataFrame(
                {'mean': [mean], 'std': [std], 'max': [max_value], 'min': [min_value], 'median': [median],
                 'kurtosis': [kurtosis], 'skew': [skew], 'rms': [rms], 'crest_factor': [crest_factor],
                 'impulse_factor': [impulse_factor], 'clearance_factor': [clearance_factor], 'label': [label]})],
                             ignore_index=True)

# 保存训练数据
data.to_csv('labeled_files.csv', index=False)
