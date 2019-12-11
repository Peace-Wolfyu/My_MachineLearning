# -*- coding: utf-8 -*-

# @Time  : 2019/12/11 14:08

# @Author : Mr.Lin

'''

看一个利用专家知识的特例——虽然在这个例子中，对这些专家知识更正确的
叫法应该是“常识”。任务是预测在 Andreas 家门口的自行车出租。
在纽约，Citi Bike 运营着一个带有付费系统的自行车租赁站网络。这些站点遍布整个城市，
提供了一种方便的交通方式。自行车出租数据以匿名形式公开（https://www.citibikenyc.
com/system-data），并用各种方法进行了分析。我们想要解决的任务是，对于给定的日期和
时间，预测有多少人将会在 Andreas 的家门口租一辆自行车——这样他就知道是否还有自
行车留给他。
我们首先将这个站点 2015 年 8 月的数据加载为一个 pandas 数据框。我们将数据重新采样
为每 3 小时一个数据，以得到每一天的主要趋势：
'''

import mglearn
import matplotlib.pyplot as plt
import pandas as pd

citibike = mglearn.datasets.load_citibike()

'''
Citi Bike data:
starttime
2015-08-01 00:00:00     3.0
2015-08-01 03:00:00     0.0
2015-08-01 06:00:00     9.0
2015-08-01 09:00:00    41.0
2015-08-01 12:00:00    39.0
Freq: 3H, Name: one, dtype: float64
'''
# print("Citi Bike data:\n{}".format(citibike.head()))


'''
下面这个示例给出了整个月租车数量的可视化
'''
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
freq='D')
plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Rentals")

# plt.show()



'''
观察此数据，我们可以清楚地区分每 24 小时中的白天和夜间。工作日和周末的模式似乎
也有很大不同。在对这种时间序列上的预测任务进行评估时，我们通常希望从过去学习并
预测未来。也就是说，在划分训练集和测试集的时候，我们希望使用某个特定日期之前的
所有数据作为训练集，该日期之后的所有数据作为测试集。这是我们通常使用时间序列预
测的方式：已知过去所有的出租数据，我们认为明天会发生什么？我们将使用前 184 个数
据点（对应前 23 天）作为训练集，剩余的 64 个数据点（对应剩余的 8 天）作为测试集
'''


'''
在我们的预测任务中，我们使用的唯一特征就是某一租车数量对应的日期和时间。因此输
入特征是日期和时间，比如 2015-08-01 00:00:00 ，而输出是在接下来 3 小时内的租车数量
（根据我们的 DataFrame ，在这个例子中是 3）。
'''


'''
在计算机上存储日期的常用方式是使用 POSIX 时间（这有些令人意外），它是从 1970 年 1
月 1 日 00:00:00（也就是 Unix 时间的起点）起至现在的总秒数。首先，我们可以尝试使用
这个单一整数特征作为数据表示：
'''

# 提取目标值（租车数量）
y = citibike.values
# 利用"%s"将时间转换为POSIX时间
X = citibike.index.strftime("%s").astype("int").reshape(-1, 1)

'''
我们首先定义一个函数，它可以将数据划分为训练集和测试集，构建模型并将结果可视化
'''



# 使用前184个数据点用于训练，剩余的数据点用于测试
n_train = 184
# 对给定特征集上的回归进行评估和作图的函数
def eval_on_features(features, target, regressor):
    # 将给定特征划分为训练集和测试集
    X_train, X_test = features[:n_train], features[n_train:]
    # 同样划分目标数组
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90,
    ha="left")
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',
    label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")




'''
我们之前看到，随机森林需要很少的数据预处理，因此它似乎很适合作为第一个模型。我
们使用 POSIX 时间特征 X ，并将随机森林回归传入我们的 eval_on_features 函数
'''



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
plt.figure()
eval_on_features(X, y, regressor)













