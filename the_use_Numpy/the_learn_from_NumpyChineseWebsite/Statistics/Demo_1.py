# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 14:21'



"""
数学和统计方法

在这些方法中，布尔值中的True和False会被相应地转换为1和0参与计算

amin(a[, axis, out, keepdims])

返回数组的最小值或沿轴的最小值

amax(a[, axis, out, keepdims])

返回数组的最大值或沿轴的最大值

nanmin(a[, axis, out, keepdims])

返回数组的最小值或沿轴的最小值，忽略任何NAN

nanmax(a[, axis, out, keepdims])

返回数组的最大值或沿轴方向的最大值，忽略任何NAN

median(a[, axis, out, overwrite_input, keepdims])

沿指定轴计算中值

average(a[, axis, weights, returned])

计算沿指定轴的加权平均

mean(a[, axis, dtype, out, keepdims])

沿指定的轴计算算术平均值

std(a[, axis, dtype, out, ddof, keepdims])

计算沿指定轴的标准偏差

var(a[, axis, dtype, out, ddof, keepdims])

计算沿指定轴的方差

nanmedian(a[, axis, out, overwrite_input, …])

在忽略NAS的情况下，沿指定的轴计算中值

nanmean(a[, axis, dtype, out, keepdims])

计算沿指定轴的算术平均值，忽略NAN

nanstd(a[, axis, dtype, out, ddof, keepdims])

计算指定轴上的标准偏差，而忽略NAN

nanvar(a[, axis, dtype, out, ddof, keepdims])

计算指定轴上的方差，同时忽略NAN
"""

import numpy as np


arr = np.array([[np.NaN, 0, 1, 2, 3],
                [ 4, 5, np.NaN, 6, 7],
                [ 8, 9, 10, np.NaN, 11]])

# print(arr)
# >>>
# [[nan  0.  1.  2.  3.]
#  [ 4.  5. nan  6.  7.]
#  [ 8.  9. 10. nan 11.]]

# np.amax(arr)
# >>>
# 空值

# print(np.nanmax(arr)) # 默认是计算flatten之后的数组，可以指定axis参数 0/1
# >>>  11.0

# print(np.nanmin(arr))
# >>>0.0

# print(np.nanmean(arr)) # 平均值
# >>>5.5

# print(np.nanmedian(arr)) # 中位数
# >>>5.5

# print(np.nanstd(arr)) # 标准差
# >>>  3.452052529534663

# print(np.nanvar(arr)) # 方差
# >>>  11.916666666666666

"""
布尔值True和False会被转为1和0参与计算

"""
bool_arr = np.array([0.7, 0, 0.5, True, False, True, True, True, False, True])

# print(bool_arr.sum())
# >>>  6.2
arr2 = np.array([0.1, 0.2, 0.3, True, False])
# print(np.amax(arr2))
# print(np.amin(arr2))
# >>>
# 1.0
# 0.0

























