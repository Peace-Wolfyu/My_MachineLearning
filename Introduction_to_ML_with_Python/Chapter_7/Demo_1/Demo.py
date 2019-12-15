# -*- coding: utf-8 -*-

# @Time  : 2019/12/12 12:56

# @Author : Mr.Lin




'''
四种类型的字符串数据：
• 分类数据
• 可以在语义上映射为类别的自由字符串
• 结构化字符串数据
• 文本数据

'''


'''
使用由斯坦福研究员 Andrew Maas 收集的 IMDb
（Internet Movie Database，互联网电影数据库）网站的电影评论数据集。 4 这个数据集包
含评论文本，还有一个标签，用于表示该评论是“正面的”（positive）还是“负面的”
（negative）。IMDb 网站本身包含从 1 到 10 的打分。为了简化建模，这些评论打分被归纳
为一个二分类数据集，评分大于等于 7 的评论被标记为“正面的”，评分小于等于 4 的评
论被标记为“负面的”，中性评论没有包含在数据集中。我们不讨论这种方法是否是一种
好的数据表示，而只是使用 Andrew Maas 提供的数据。
将数据解压之后，数据集包括两个独立文件夹中的文本文件，一个是训练数据，一个是测
试数据。每个文件夹又都有两个子文件夹，一个叫作 pos，一个叫作 neg：
'''


'''
pos 文件夹包含所有正面的评论，每条评论都是一个单独的文本文件，neg 文件夹与之类
似。 scikit-learn 中有一个辅助函数可以加载用这种文件夹结构保存的文件，其中每个子
文件夹对应于一个标签，这个函数叫作 load_files 。我们首先将 load_files 函数应用于训
练数据
'''

from sklearn.datasets import load_files
reviews_train = load_files("train/")
# load_files返回一个Bunch对象，其中包含训练文本和训练标签
text_train, y_train = reviews_train.data, reviews_train.target

'''
type of text_train: <class 'list'>
length of text_train: 75000
text_train[1]:
b"Amount of disappointment I am getting these days seeing movies like Partner, Jhoom Barabar and now, Heyy Babyy is gonna end my habit of seeing first day shows.<br /><br />The movie is an utter disappointment because it had the potential to become a laugh riot only if the d\xc3\xa9butant director, Sajid Khan hadn't tried too many things. Only saving grace in the movie were the last thirty minutes, which were seriously funny elsewhere the movie fails miserably. First half was desperately been tried to look funny but wasn't. Next 45 minutes were emotional and looked totally artificial and illogical.<br /><br />OK, when you are out for a movie like this you don't expect much logic but all the flaws tend to appear when you don't enjoy the movie and thats the case with Heyy Babyy. Acting is good but thats not enough to keep one interested.<br /><br />For the positives, you can take hot actresses, last 30 minutes, some comic scenes, good acting by the lead cast and the baby. Only problem is that these things do not come together properly to make a good movie.<br /><br />Anyways, I read somewhere that It isn't a copy of Three men and a baby but I think it would have been better if it was."
'''
# print("type of text_train: {}".format(type(text_train)))
# print("length of text_train: {}".format(len(text_train)))
# print("text_train[1]:\n{}".format(text_train[1]))

'''
可以看到， text_train 是一个长度为 25 000 的列表
6 ，其中每个元素是包含一条评论的字
符串。我们打印出索引编号为 1 的评论。你还可以看到，评论中包含一些 HTML 换行符
（ <br /> ）。虽然这些符号不太可能对机器学习模型产生很大影响，但最好在继续下一步之
前清洗数据并删除这种格式：
'''


'''
text_train 的元素类型与你所使用的 Python 版本有关。在 Python 3 中，它们是 bytes 类
型，是表示字符串数据的二进制编码。在 Python 2 中， text_train 包含的是字符串。
'''
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]


'''
收集数据集时保持正类和反类的平衡，这样所有正面字符串和负面字符串的数量相等：
'''
import numpy as np

'''
Samples per class (training): [12500 12500 50000]

'''
# print("Samples per class (training): {}".format(np.bincount(y_train)))



'''
用同样的方式加载测试数据集：
'''

reviews_test = load_files("test/")
text_test, y_test = reviews_test.data, reviews_test.target
'''
Number of documents in test data: 25000
Samples per class (test): [12500 12500]
'''
# print("Number of documents in test data: {}".format(len(text_test)))
# print("Samples per class (test): {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]





















































































































































