# -*- coding: utf-8 -*-
# @Time  : 2019/12/22 14:25
# @Author : Mr.Lin

'''

加载MNIST数据集，这是一组由美国高中生和人口调查局员工手写的70000个数字的图片。每张图像都用其代表的数字标记。

'''

from sklearn.datasets import fetch_mldata

# 类型为  <class 'sklearn.utils.Bunch'>
mnist = fetch_mldata('MNIST Original',data_home='./')


