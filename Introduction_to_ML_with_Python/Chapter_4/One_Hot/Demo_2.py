# -*- coding: utf-8 -*-
# @Time  : 2019/12/10 20:28
# @Author : Mr.Lin


'''

数字可以编码分类变量

'''
from IPython.core.display import display

'''
在 adult 数据集的例子中，分类变量被编码为字符串。一方面，可能会有拼写错误；但另
一方面，它明确地将一个变量标记为分类变量。无论是为了便于存储还是因为数据的收集
方式，分类变量通常被编码为整数。例如，假设 adult 数据集中的人口普查数据是利用问
卷收集的， workclass 的回答被记录为 0（在第一个框打勾）、1（在第二个框打勾）、2（在
第三个框打勾），等等。现在该列包含数字 0 到 8，而不是像 "Private" 这样的字符串。如
果有人观察表示数据集的表格，很难一眼看出这个变量应该被视为连续变量还是分类变量。但是，如果知道这些数字表示的是就业状况，那么很明显它们是不同的状态，不应该
用单个连续变量来建模。
'''



'''
pandas 的 get_dummies 函数将所有数字看作是连续的，不会为其创建虚拟变量。为了解决
这个问题，你可以使用 scikit-learn 的 OneHotEncoder ，指定哪些变量是连续的、哪些变
量是离散的，你也可以将数据框中的数值列转换为字符串。为了说明这一点，我们创建一
个两列的 DataFrame 对象，其中一列包含字符串，另一列包含整数：
'''

import pandas as pd
# 创建一个DataFrame，包含一个整数特征和一个分类字符串特征
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
'Categorical Feature': ['socks', 'fox', 'socks', 'box']})

'''
  Categorical Feature  Integer Feature
0               socks                0
1                 fox                1
2               socks                2
3                 box                1
'''
# display(demo_df)

'''
使用 get_dummies 只会编码字符串特征，不会改变整数特征
'''
pd.get_dummies(demo_df)

display(demo_df)
































