# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/12 16:49'


"""
索引与切片
"""

import numpy as np

"""
针对一维数组
"""

arr = np.arange(10)

print(arr)

print(arr[5])

print(arr[5:9])


"""
二维数组
"""

# 写法是个细节
arr2d = np.array(
    [
    [1,2,3],
    [4,5,6],
    [7,8,9]
        ]
)

print(arr2d[1])

print(arr2d[0][2])
print(arr2d[0,2])


"""
2*2*3数组
"""

arr3d = np.array(
    [
        [
            [1,2,3],
            [4,5,6]
        ],
        [
            [7, 8, 9],
            [10, 11, 12]
        ]

    ]

)

print(arr3d)


# 2*3数组
print(arr3d[0])


"""
花式索引
"""


arr_1 = np.empty((8,4))

for i in range(8):
    arr_1[i] = i

print(arr_1)

"""
为了以特定顺序选取子集
只需要传入一个用于指定顺序的整数列表或者ndarray即可
"""

print(arr_1[[4,1,0,6]])
