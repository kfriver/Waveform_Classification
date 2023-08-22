# Waveform_Classification
A program to classify ini waveform files by their waveform feature.

how to use

1.Use the already trained model directly
① Place your ini waveform data file in the data_origin directory.
② Run the cls_model.py file and view the results in classified_data.
③ If you want to delete the preview image of a categorized wave file, you can run del_png.py.

2.Perform your own model training
① Put the already categorized data to be trained (only ini files are needed) in the folder 0, 1 in the data directory. Note: 0 is a low noise waveform and 1 is a high noise waveform.
② Running label2csv.py generates labeled_files.csv, a file containing the waveform feature label data for all ini files.
③ Run train.py and wait for some time to check the evaluation results of the model to get the model file classifier.joblib.
④ Do the "1.Use the already trained model directly" all over again.



###########################################################################################################################################################################



# 波形分类
根据波形特征对 ini 波形文件进行分类的程序。

使用方法

1.直接使用已训练好的模型
将 ini 波形数据文件放在 data_origin 目录下。
运行 cls_model.py 文件并在 classified_data 中查看结果。
③ 如果要删除已分类波形文件的预览图像，可以运行 del_png.py。

2.执行自己的模型训练
将已分类的待训练数据（只需 ini 文件）放入数据目录下的 0、1 文件夹。注：0 为低噪声波形，1 为高噪声波形。
运行 label2csv.py 会生成 labeled_files.csv，该文件包含所有 ini 文件的波形特征标签数据。
③ 运行 train.py，等待一段时间检查模型的评估结果，得到模型文件 classifier.joblib。
④ 重新执行 "1.直接使用已训练好的模型"。

