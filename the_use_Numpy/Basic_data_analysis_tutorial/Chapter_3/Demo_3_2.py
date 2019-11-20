# -*- coding: utf-8 -*-
# @Time  : 2019/11/20 21:21
# @Author : Mr.Lin

"""

读写文件

"""
import numpy as np

i2 = np.eye(2)
print(i2)
# >>>
# [[1. 0.]
#  [0. 1.]]

'将数据保存到文件'

np.savetxt("eye.txt",i2)
'在当前目录出现了定义的文件'























