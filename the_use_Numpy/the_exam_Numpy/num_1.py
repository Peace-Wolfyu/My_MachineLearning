# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/13 15:58'


"""
 1 Import the numpy package under the name `np` 
 通用引用别名‘np’导入numpy包
"""

import numpy as np


"""
#### 2. Print the numpy version and the configuration (★☆☆)

打印numpy的版本以及配置
"""

print("numpy 版本")
print(np.__version__)
print("numpy 配置")
np.show_config()


"""
#### 3. Create a null vector of size 10 (★☆☆)

创建一个空的 大小为10 的向量
"""
print("************************************")

# zeros()  知识点
z = np.zeros(10)
print(z)

"""
#### 4.  How to find the memory size of any array (★☆☆)

输出任意数组的内存大小
"""
print("************************************")

z_1 = np.zeros((10,10))
print(z_1)
#  size 知识点  itemsize 知识点
print("%d bytes" % (z_1.size * z_1.itemsize))


"""
#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

通过命令行获取numpy add 函数的 文档

"""

"""
%run `python -c "import numpy; numpy.info(numpy.add)"`
"""

"""
#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

创建一个空的 大小为10 的向量，但是第五个的值为 1
"""

z_2 = np.zeros(10)

# 索引从 0 开始
z_2[4] = 1

print(z_2)

"""
#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

创建一个向量  里面的值 从 10 到 49
"""


print("************************************")

# arange() 知识点
z_3 = np.arange(10,50)

print(z_3)


"""
#### 8.  Reverse a vector (first element becomes last) (★☆☆)

转置一个向量
"""
print("************************************")

z_4 = np.arange(10)

print(z_4)

# [::-1]知识点
z_4 = z_4[::-1]

print(z_4)

"""
#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

创建一个 3 * 3 的矩阵  数值设置为 从 0 到 8
"""
print("************************************")

z_5 = np.arange(9).reshape(3,3)

print(z_5)

"""

10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

从 给出列表中找出 不为 0 的 元素的索引

"""
print("************************************")

# nonzero() 知识点
z_6 = np.nonzero([1,2,0,0,4,0])
print(z_6)




