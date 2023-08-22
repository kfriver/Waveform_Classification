# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 15:16
# @Author  : longfei Kang
# @File    : del_png.py

import os


def delete_png_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                os.remove(os.path.join(root, file))


# 输入需要删除的目录
delete_png_files("./classified_data")
