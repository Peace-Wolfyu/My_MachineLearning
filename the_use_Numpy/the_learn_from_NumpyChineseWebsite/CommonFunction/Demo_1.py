# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/16 15:46'



"""
通用函数

通用函数（或简称为ufunc）是一种以元素为单位对ndarray进行操作的函数，支持数组广播，类型转换和其他一些标准特性。 也就是说，ufunc是一个函数的“矢量化”包装器，它接受固定数量的特定输入并产生固定数量的特定输出
在NumPy中，通用函数是numpy.ufunc类的实例

https://docs.scipy.org/doc/numpy/reference/ufuncs.html

一元ufunc


函数	说明
abs、fabs	计算整数、浮点数或复数的绝对值。对于非复数值，可以使用更快的fabs
sqrt	计算各元素的平方根。相当于arr ** 0.5
square	计算各元素的平方。相当于arr ** 2
exp	计算各元素的指数。相当于e的x次方
log、log10、log2、log1p	分别对应自然对数（底数为e）、底数为10的log、底数为2的log、log(1 + p)
sign	计算各元素的正负号：1（正数）、0（零）、-1（负数）
ceil	计算各元素的ceiling值，即大于等于该值的最小整数
floor	计算各元素的floor值。即小于等于该值的最大整数
rint	将各元素值四舍五入到最接近的整数，保留dtype
modf	将数组的小数和整数部分以两个独立数组的形式返回
isnan	返回一个表示“哪些值是NaN”的布尔型数组
isfinite、isinf	分别返回一个表示“哪些元素是有穷的（非inf，非NaN）”或“哪些元素是无穷的”的布尔型数组
cos、cosh、sin、sinh、tan、tanh	普通型和双曲型三角函数
arccos、arccosh、arcsin、arcsinh、arctan、arctanh	反三角函数
logical_not	计算个元素的not x的真值。相当于-arr


二元ufunc

函数	说明
add	将数组中对应的元素相加
subtract	从第一个数组中减去第二个数组中的元素
multiply	将数组元素相乘
divide、floor_divide	除法或向下取整除法（丢弃余数）
power	对第一个数组中的元素A，根据第二个数组中的相应元素B，计算A的B次方
maximum、fmax	元素级的最大值计算。fmax将忽略NaN
minimum、fmin	元素级的最小值计算。fmin将忽略NaN
mod	元素级别的求模计算（除法的余数）
copysign	将第二个数组中的值得符号复制给第一个数组中的值
greater、greater_equal、less、less_equal、equal、not_equal、	执行元素级的比较运算，最终产生布尔数组。相当于中缀运算符号>、>=、<、<=、==、!=
logical_and、logical_or、logical_xor	执行元素级的真值逻辑运算。相当于中缀运算符&、| 、^

"""
import numpy as np

arr1 = np.array([ 2.06071314, -0.02325376,  0.34784075, -0.92764143, -0.37922931,
       -1.22737788, -1.54119057,  0.03776073,  0.47776534, -0.33724855])

# print(arr1)
# >>>
# [ 2.06071314 -0.02325376  0.34784075 -0.92764143 -0.37922931 -1.22737788
#  -1.54119057  0.03776073  0.47776534 -0.33724855]

# print(np.abs(arr1))
# >>>
# [2.06071314 0.02325376 0.34784075 0.92764143 0.37922931 1.22737788
#  1.54119057 0.03776073 0.47776534 0.33724855]


arr2 = np.array([4, 9, 16, 25])
# print(np.sqrt(arr2))
# >>>
# [2. 3. 4. 5.]

# print(np.sign(arr1)) # 正负号)
# >>>
# [ 1. -1.  1. -1. -1. -1. -1.  1.  1. -1.]

arr3 = np.array([1, 2, 3, np.NaN, 4, np.NaN])
# print(np.isnan(arr3))
# >>>
# [False False False  True False  True]

arr4 = np.array([1, 2, 3])
arr5 = np.array([4, 5, 6])
# print(np.add(arr4, arr5))
# >>>
# [5 7 9]

# print(np.mod(arr2, 3)) # 求模)
# >>>
# [1 0 1 1]












































