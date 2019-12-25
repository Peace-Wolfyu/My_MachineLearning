# -*- coding: utf-8 -*-
# @Time  : 2019/12/25 20:06
# @Author : Mr.Lin

'''

泰坦尼克号数据处理

'''

import pandas as pd

# 首先加载数据
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 接下来进行数据预处理

# 将年龄这一列的数据缺失值进行填充

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

# 删除掉缺失值太多的列，与预测结果无关的列

train_data = train_data.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)

# 字符数据转换成数值
train_data["Sex"] = (train_data["Sex"] == 'male').astype(int)
ls = train_data["Embarked"].unique().tolist()
train_data["Embarked"] = train_data["Embarked"].apply(lambda x: ls.index(x))

# 测试集和训练集划分

X = train_data.loc[:, train_data.columns != 'Survived']
y = train_data.loc[:, train_data.columns == 'Survived']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)

for i in [Xtrain, Xtest, Ytrain, Ytest]:  # 重排序号，使之有序
    i.index = range(i.shape[0])

# 训练模型


clf = DecisionTreeClassifier(random_state=20
                             , criterion='gini'
                             , max_depth=4
                             , min_samples_leaf=1
                             , splitter='random'
                             )
clf = clf.fit(Xtrain, Ytrain)



print(cross_val_score(clf, Xtest, Ytest, cv=3, scoring="accuracy"))






# 读取测试集数据

test = pd.read_csv('test.csv')


# 把测试集预处理操作封装
def clean_data(data):
    data = data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId']
                     , axis=1
                     )
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())  #
    data = data.dropna(axis=0)
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return data


test_data = clean_data(test)

clf.predict(test_data)
