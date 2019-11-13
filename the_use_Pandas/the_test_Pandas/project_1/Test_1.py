# -*- coding: utf-8 -*-
# @Time  : 2019/11/13 21:08
# @Author : Mr.Lin


import pandas as pd

chipo=pd.read_csv('C:\\Users\\linziyu\\Desktop\\data\\chipotle.tsv',sep='\t')


# 查看前十行内容
# chipo.head(10)


# print(chipo.head(10))


# 数据集中有多少个列(columns)

print(chipo.shape[1])
print(" ")
# 打印出全部的列名称
print(chipo.columns)
print(" ")
# 数据集的索引是怎样的

print(chipo.index)
print(" ")
# 被下单数最多商品(item)是什么?


# 运行以下代码，做了修正
c = chipo[['item_name','quantity']].groupby(['item_name'],as_index=False).agg({'quantity':sum})
c.sort_values(['quantity'],ascending=False,inplace=True)
c.head()
print(c.head())
print(" ")
# 在item_name这一列中，一共有多少种商品被下单？

unique = chipo['item_name'].nunique()
print(unique)
print(" ")

# 在choice_description中，下单次数最多的商品是什么？

# 运行以下代码，存在一些小问题
chipo['choice_description'].value_counts().head()

print(" ")

# 一共有多少商品被下单？

total_items_orders = chipo['quantity'].sum()
print(total_items_orders)
print(" ")
print(" ")


# 将item_price转换为浮点数
dollarizer = lambda x: float(x[1:-1])
chipo['item_price'] = chipo['item_price'].apply(dollarizer)
print(chipo['item_price'])
print(" ")

print(" ")

# 在该数据集对应的时期内，收入(revenue)是多少

chipo['sub_total'] = round(chipo['item_price'] * chipo['quantity'],2)
sum = chipo['sub_total'].sum()
print(sum)
print(" ")

# 在该数据集对应的时期内，一共有多少订单？

unique_order = chipo['order_id'].nunique()
print(unique_order)
print(" ")


# 每一单(order)对应的平均总价是多少？

# 运行以下代码，已经做过更正
mean_order = chipo[['order_id','sub_total']].groupby(by=['order_id']
).agg({'sub_total':'sum'})['sub_total'].mean()
print(mean_order)
print(" ")
















































































