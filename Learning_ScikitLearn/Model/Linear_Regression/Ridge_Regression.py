# -*- coding: utf-8 -*-

# @Time  : 2019/12/16 13:50

# @Author : Mr.Lin

'''


岭回归
岭回归也是一种用于回归的线性模型，因此它的预测公式与普通最小二乘法相同。但在岭
回归中，对系数（w）的选择不仅要在训练数据上得到好的预测结果，而且还要拟合附加
约束。我们还希望系数尽量小。换句话说，w 的所有元素都应接近于 0。直观上来看，这
意味着每个特征对输出的影响应尽可能小（即斜率很小），同时仍给出很好的预测结果。
这种约束是所谓正则化（regularization）的一个例子。正则化是指对模型做显式约束，以
监督学习 ｜ 39
避免过拟合。岭回归用到的这种被称为 L2 正则化

'''
from sklearn.cross_validation import cross_val_predict
import matplotlib.pyplot as plt

from Learning_ScikitLearn.Model.Linear_Regression.DataSource import X_train,X_test,y_train,y_test,data_X,data_y
from sklearn.linear_model import Ridge
from sklearn import metrics


ridge_model = Ridge()

ridge_model.fit(X_train,y_train)

y_prediction = ridge_model.predict(X_test)

print(y_prediction)
print("")

'''
MSE: 24.922349436023072

'''
print("MSE:", metrics.mean_squared_error(y_test, y_prediction))

print("")

# 交叉验证

predicted = cross_val_predict(ridge_model, data_X, data_y, cv=10)

print("")

# MSE: 33.96300084613328

print("MSE:", metrics.mean_squared_error(data_y, predicted))

plt.scatter(data_y, predicted, color='y', marker='o')
plt.scatter(data_y, data_y,color='g', marker='+')
# plt.show()


'''
Ridge 模型在模型的简单性（系数都接近于 0）与训练集性能之间做出权衡。简单性和训练
集性能二者对于模型的重要程度可以由用户通过设置 alpha 参数来指定。在前面的例子中，
我们用的是默认参数 alpha=1.0 。但没有理由认为这会给出最佳权衡。 alpha 的最佳设定
值取决于用到的具体数据集。增大 alpha 会使得系数更加趋向于 0，从而降低训练集性能，
但可能会提高泛化性能。

'''
print("")
print("")
def test_alpha_effect():
    alphas = [0.1,1,10]

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)

        ridge.fit(X_train,y_train)

        y_pre = ridge.predict(X_test)

        print("alpha 为: {} 下的MSE ： {} \n".format(alpha,metrics.mean_squared_error(y_test, y_pre)))


# alpha 为: 0.1 下的MSE ： 27.621569581874475
#
# alpha 为: 1 下的MSE ： 27.961634975902424
#
# alpha 为: 10 下的MSE ： 27.974375820695315

test_alpha_effect()

















































