# -*- coding: utf-8 -*-

# @Time  : 2019/11/30 16:15

# @Author : ly

import Learning_From_Data.Chapter_1.exercise_1_4.Helper as help
import numpy as np
"""
对 1.4算法进行测试
"""

#设置随机种子，保证每次结果一致
seed = 42
rnd = np.random.RandomState(seed)
N = 20
d = 2
a, b, c, X, y, s, w = help.f(N, d, rnd)
help.plot_helper(a, b, c, X, y, s, w)










