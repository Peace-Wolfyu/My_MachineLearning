# -*- coding: utf-8 -*-
# @Time  : 2019/12/10 19:52
# @Author : Mr.Lin

'''

到目前为止，表示分类变量最常用的方法就是使用 one-hot 编码（one-hot-encoding）或
N 取一编码（one-out-of-N encoding），也叫虚拟变量（dummy variable）。虚拟变量背后的
思想是将一个分类变量替换为一个或多个新特征，新特征取值为 0 和 1。对于线性二分类
（以及 scikit-learn 中其他所有模型）的公式而言，0 和 1 这两个值是有意义的，我们可
以像这样对每个类别引入一个新特征，从而表示任意数量的类别。
'''


'''
比如说， workclass 特征的可能取值包括 "Government Employee" 、 "Private Employee" 、
"Self Employed" 和 "Self Employed Incorporated" 。为了编码这 4 个可能的取值，我
们创建了 4 个新特征，分别叫作 "Government Employee" 、 "Private Employee" 、 "Self
Employed" 和 "Self Employed Incorporated" 。如果一个人的 workclass 取某个值，那么对
应的特征取值为 1，其他特征均取值为 0。因此，对每个数据点来说，4 个新特征中只有一
个的取值为 1。这就是它叫作 one-hot 编码或 N 取一编码的原因。
'''

'''
将数据转换为分类变量的 one-hot 编码有两种方法：一种是使用 pandas ，一种是使用
scikit-learn 。在写作本书时，使用 pandas 要稍微简单一些，所以我们选择这种方法。首
先，我们使用 pandas 从逗号分隔值（CSV）文件中加载数据
'''

import pandas as pd
from IPython.display import  display

# 文件中没有包含列名称的表头，因此我们传入header=None
# 然后在"names"中显式地提供列名称
data = pd.read_csv(
"adult.data", header=None, index_col=False,
names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'])
# 为了便于说明，我们只选了其中几列
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
'occupation', 'income']]
# IPython.display可以在Jupyter notebook中输出漂亮的格式
'''
   age          workclass   education   gender  hours-per-week  \
0   39          State-gov   Bachelors     Male              40   
1   50   Self-emp-not-inc   Bachelors     Male              13   
2   38            Private     HS-grad     Male              40   
3   53            Private        11th     Male              40   
4   28            Private   Bachelors   Female              40   

           occupation  income  
0        Adm-clerical   <=50K  
1     Exec-managerial   <=50K  
2   Handlers-cleaners   <=50K  
3   Handlers-cleaners   <=50K  
4      Prof-specialty   <=50K  
'''
# display(data.head())

'''
检查字符串编码的分类数据
'''
'''
读取完这样的数据集之后，最好先检查每一列是否包含有意义的分类数据。在处理人工
（比如网站用户）输入的数据时，可能没有固定的类别，拼写和大小写也存在差异，因此
可能需要预处理。举个例子，有人可能将性别填为“male”（男性），有人可能填为“man”
（男人），而我们希望能用同一个类别来表示这两种输入。检查列的内容有一个好方法，就
是使用 pandas Series （ Series 是 DataFrame 中单列对应的数据类型）的 value_counts 函
数，以显示唯一值及其出现次数：
'''

'''
 Male      21790
 Female    10771
Name: gender, dtype: int64
'''

'''
可以看到，在这个数据集中性别刚好有两个值： Male 和 Female ，这说明数据格式已经很
好，可以用 one-hot 编码来表示。在实际的应用中，你应该查看并检查所有列的值
'''
# print(data.gender.value_counts())


'''
用 pandas 编码数据有一种非常简单的方法，就是使用 get_dummies 函数。 get_dummies 函
数自动变换所有具有对象类型（比如字符串）的列或所有分类的列（这是 pandas 中的一个
特殊概念）
'''


'''
Original features:
 ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income'] 

Features after get_dummies:
 ['age', 'hours-per-week', 'workclass_ ?', 'workclass_ Federal-gov', 'workclass_ Local-gov',
  'workclass_ Never-worked', 'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc',
   'workclass_ State-gov', 'workclass_ Without-pay', 'education_ 10th', 'education_ 11th', 'education_ 12th',
    'education_ 1st-4th', 'education_ 5th-6th', 'education_ 7th-8th', 'education_ 9th', 'education_ Assoc-acdm', 
    'education_ Assoc-voc', 'education_ Bachelors', 'education_ Doctorate', 'education_ HS-grad', 'education_ Masters', 
    'education_ Preschool', 'education_ Prof-school', 'education_ Some-college', 'gender_ Female', 'gender_ Male', 'occupation_ ?', 
    'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 
    'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty',
     'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'income_ <=50K', 'income_ >50K']
'''

'''
连续特征 age 和 hours-per-week 没有发生变化，而分类特征的每个可能取值
都被扩展为一个新特征：
'''
# print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
# print("Features after get_dummies:\n", list(data_dummies.columns))

# print(data_dummies.head())


'''
下面我们可以使用 values 属性将 data_dummies 数据框（ DataFrame ）转换为 NumPy 数组，
然后在其上训练一个机器学习模型。在训练模型之前，注意要把目标变量（现在被编码为
两个 income 列）从数据中分离出来。将输出变量或输出变量的一些导出属性包含在特征表
示中，这是构建监督机器学习模型时一个非常常见的错误。
'''



'''
在这个例子中，我们仅提取包含特征的列，也就是从 age 到 occupation_ Transport-moving
的所有列。这一范围包含所有特征，但不包含目标
'''



features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
# 提取NumPy数组
X = features.values
y = data_dummies['income_ >50K'].values

'''
X.shape: (32561, 44) y.shape: (32561,)


现在数据的表示方式可以被 scikit-learn 处理
'''
# print("X.shape: {} y.shape: {}".format(X.shape, y.shape))



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
































