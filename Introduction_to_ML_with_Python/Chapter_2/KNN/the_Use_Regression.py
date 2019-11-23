# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/23 13:24'



" 用于回归的k近邻算法"

from  sklearn.neighbors import KNeighborsRegressor
import mglearn
from sklearn.model_selection import train_test_split


X,y = mglearn.datasets.make_wave(n_samples = 40)

# 将数据集分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# 模型实例化 并将邻居个数设置为3
reg = KNeighborsRegressor(n_neighbors=3)

# 利用训练数据和训练目标值来拟合数据
reg.fit(X_train,y_train)

# 对测试机进行预测
print("Test set prediction : \n{}".format(reg.predict(X_test)))
#
# >>>
# Test set prediction :
# [-0.05396539  0.35686046  1.13671923 -1.89415682 -1.13881398 -1.63113382
#   0.35686046  0.91241374 -0.44680446 -1.13881398]


'用score方法来评估模型'

print("Test set R2 : {:.2f}".format(reg.score(X_test,y_test)))

# >>>
# Test set R2 : 0.83














































































