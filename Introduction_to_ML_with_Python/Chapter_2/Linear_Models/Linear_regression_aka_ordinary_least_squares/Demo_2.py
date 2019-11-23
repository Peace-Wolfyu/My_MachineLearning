# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/23 14:03'



import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

" 针对更加复杂的数据集合" \
"波士顿房价数据集 有506个样本以及105个导出特征"


X,y = mglearn.datasets.load_extended_boston()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

lr = LinearRegression().fit(X_train,y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# >>>
# Training set score: 0.95
# Test set score: 0.61


























