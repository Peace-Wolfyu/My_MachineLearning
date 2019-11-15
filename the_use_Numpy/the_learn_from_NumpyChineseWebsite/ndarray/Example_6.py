# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/15 15:57'

"""
迭代数组
nditerNumPy 1.6中引入的迭代器对象提供了许多灵活的方法来以系统的方式访问一个或多个数组的所有元素。 本页介绍了在Python中使用该对象进行数组计算的一些基本方法，然后总结了如何在Cython中加速内部循环。 由于Python暴露 nditer 是C数组迭代器API的相对简单的映射， 因此这些想法​​还将提供有关使用C或C ++的数组迭代的帮助
"""


import numpy as np

# 单数组迭代
# 可以使用的最基本任务nditer是访问数组的每个元素。使用标准Python迭代器接口逐个提供每个元素。
#
# 示例：

a = np.arange(6).reshape(2,3)
for x in np.nditer(a):
    # print(x, end=' ')
    pass

# >>>
# 0 1 2 3 4 5

# 对于此迭代，需要注意的重要一点是，
# 选择顺序以匹配数组的内存布局，而不是使用标准C或Fortran排序。
# 这样做是为了提高访问效率，反映了默认情况下只需要访问每个元素而不关心特定排序的想法。
# 我们可以通过迭代前一个数组的转置来看到这一点，而不是以C顺序获取该转置的副本



a1 = np.arange(6).reshape(2,3)
for x in np.nditer(a.T):
     # print(x, end=' ')
    pass

# >>>
# 0 1 2 3 4 5

for y in np.nditer(a1.T.copy(order='C')):
     # print(y, end=' ')
    pass
# >>>
# 0 3 1 4 2 5

# a 和 aT 的元素以相同的顺序遍历，
# 即它们存储在内存中的顺序， 而 a.T.copy(order='C')
# 的元素以不同的顺序访问，因为它们被放入不同的内存中布局



# 控制迭代顺序
# 有时，无论内存中元素的布局如何，以特定顺序访问数组的元素都很重要。
# 该nditer对象提供了一个 命令 参数来控制迭代的这个方面。
# 具有上述行为的默认值是order ='K'以保持现有订单。 对于C顺序，
# 可以使用order ='C'覆盖它，对于Fortran顺序，可以使用order ='F'覆盖它。

a2 = np.arange(6).reshape(2,3)
for x in np.nditer(a2, order='F'):
     # print(x, end=' ')
        pass
# >>>
# 0 3 1 4 2 5
print("")
for x in np.nditer(a2.T, order='C'):
     # print(x, end=' ')
        pass
# >>>
# 0 3 1 4 2 5

# 修改数组值
# 默认情况下，nditer将输入操作数视为只读对象。 为了能够修改数组元素，
# 必须使用 'readwrite' 或 'writeonly' 每操作数标志指定读写或只写模式。
#
# 然后，nditer将生成可写的缓冲区数组，您可以修改它们。 但是，因为一旦迭代完成，
# nditer必须将此缓冲区数据复制回原始数组， 所以必须通过两种方法之一发出迭代结束时的信号。你可以：
#
# 使用 with 语句将nditer用作上下文管理器，并在退出上下文时写回临时数据。
# 完成迭代后调用迭代器的 close 方法，这将触发回写。
# 一旦调用 close 或退出其上下文，就不能再迭代nditer 。


a3 = np.arange(6).reshape(2,3)
# print(a3)
# >>>
# [[0 1 2]
#  [3 4 5]]
print("")
with np.nditer(a3, op_flags=['readwrite']) as it:
    for x in it:
        x[...] = 2 * x
# print(a3)

# >>>

# [[ 0  2  4]
#  [ 6  8 10]]




# 使用外部循环
# 在所有实施例中，到目前为止，的元素 一个 由迭代器一次一个地提供， 因为所有的循环逻辑是内部的迭代器。虽然这很简单方便，但效率不高。 更好的方法是将一维最内层循环移动到迭代器外部的代码中。 这样，NumPy的矢量化操作可以用在被访问元素的较大块上。
#
# 该nditer会尽量提供尽可能大的内循环块。 通过强制'C'和'F'顺序，我们得到不同的外部循环大小。通过指定迭代器标志来启用此模式。
#
# 观察到默认情况下保持本机内存顺序，迭代器能够提供单个一维块，而在强制Fortran命令时，它必须提供三个块，每个块包含两个元素。


a4 = np.arange(6).reshape(2,3)
for x in np.nditer(a4, flags=['external_loop']):
     # print(x, end=' ')
    pass
# >>>
# [0 1 2 3 4 5]
print("")
for x in np.nditer(a4, flags=['external_loop'], order='F'):
     # print(x, end=' ')
        pass
# >>>
# [0 3] [1 4] [2 5]


# 跟踪索引或多索引
# 在迭代期间，您可能希望在计算中使用当前元素的索引。 例如，您可能希望以内存顺序访问数组的元素，但使用C顺序，Fortran顺序或多维索引来查找不同数组中的值。
#
# Python迭代器协议没有从迭代器查询这些附加值的自然方法， 因此我们引入了一个替代语法来迭代nditer。 此语法显式使用迭代器对象本身，因此在迭代期间可以轻松访问其属性。使用此循环结构，可以通过索引到迭代器来访问当前值，并且正在跟踪的索引是属性 索引 或 multi_index， 具体取决于请求的内容。
#
# 遗憾的是，Python交互式解释器在循环的每次迭代期间打印出while循环内的表达式的值。我们使用这个循环结构修改了示例中的输出，以便更具可读性。


a5 = np.arange(6).reshape(2,3)
it = np.nditer(a5, flags=['f_index'])
while not it.finished:
     # print("%d <%d>" % (it[0], it.index), end=' ')
     it.iternext()
     pass


# >>>
# 0 <0> 1 <2> 2 <4> 3 <1> 4 <3> 5 <5>


it = np.nditer(a5, flags=['multi_index'])
while not it.finished:
     print("%d <%s>" % (it[0], it.multi_index), end=' ')
     it.iternext()

# >>>
# 0 <(0, 0)> 1 <(0, 1)> 2 <(0, 2)> 3 <(1, 0)> 4 <(1, 1)> 5 <(1, 2)>






















