# -*- coding: utf-8 -*-

# @Time  : 2019/12/12 14:11

# @Author : Mr.Lin

'''

多个单词的词袋（n元分词）
使用词袋表示的主要缺点之一是完全舍弃了单词顺序。因此，“it’s bad, not good at all”（电
影很差，一点也不好）和“it’s good, not bad at all”（电影很好，还不错）这两个字符串的
词袋表示完全相同，尽管它们的含义相反。将“not”（不）放在单词前面，这只是上下文
很重要的一个例子（可能是一个极端的例子）。幸运的是，使用词袋表示时有一种获取上
下文的方法，就是不仅考虑单一词例的计数，而且还考虑相邻的两个或三个词例的计数。
两个词例被称为二元分词（bigram），三个词例被称为三元分词（trigram），更一般的词例
序列被称为 n 元分词（n-gram）。我们可以通过改变 CountVectorizer 或 TfidfVectorizer
的 ngram_range 参数来改变作为特征的词例范围。 ngram_range 参数是一个元组，包含要考
虑的词例序列的最小长度和最大长度。下面是在之前用过的玩具数据上的一个示例：
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

bards_words =["The fool doth think he is wise,",
"but the wise man knows himself to be a fool"]


'''
默认情况下，为每个长度最小为 1 且最大为 1 的词例序列（或者换句话说，刚好 1 个词
例）创建一个特征——单个词例也被称为一元分词（unigram）
'''

# cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)
'''
Vocabulary size: 13
Vocabulary:
['be', 'but', 'doth', 'fool', 'he', 'himself', 'is', 'knows', 'man', 'the', 'think', 'to', 'wise']
'''
# print("Vocabulary size: {}".format(len(cv.vocabulary_)))
# print("Vocabulary:\n{}".format(cv.get_feature_names()))


'''
要想仅查看二元分词（即仅查看由两个相邻词例组成的序列），可以将 ngram_range 设置
为 (2, 2) ：
'''

# cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
'''
Vocabulary size: 14
Vocabulary:
['be fool', 'but the', 'doth think', 'fool doth', 'he is', 'himself to', 'is wise', 'knows himself', 'man knows', 'the fool', 'the wise', 'think he', 'to be', 'wise man']
'''
# print("Vocabulary size: {}".format(len(cv.vocabulary_)))
# print("Vocabulary:\n{}".format(cv.get_feature_names()))

'''
使用更长的词例序列通常会得到更多的特征，也会得到更具体的特征。 bard_words 的两个
短语中没有相同的二元分词
'''


'''
Transformed data (dense):
[[0 0 1 1 1 0 1 0 0 1 0 1 0 0]
 [1 1 0 0 0 1 0 1 1 0 1 0 1 1]]
'''
# print("Transformed data (dense):\n{}".format(cv.transform(bards_words).toarray()))

'''
对于大多数应用而言，最小的词例数量应该是 1，因为单个单词通常包含丰富的含义。在
大多数情况下，添加二元分词会有所帮助。添加更长的序列（一直到五元分词）也可能有
所帮助，但这会导致特征数量的大大增加，也可能会导致过拟合，因为其中包含许多非常
具体的特征。原则上来说，二元分词的数量是一元分词数量的平方，三元分词的数量是一
元分词数量的三次方，从而导致非常大的特征空间。在实践中，更高的 n 元分词在数据中
的出现次数实际上更少，原因在于（英语）语言的结构，不过这个数字仍然很大。
下面是在 bards_words 上使用一元分词、二元分词和三元分词的结果：
'''

cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
# print("Vocabulary size: {}".format(len(cv.vocabulary_)))
# print("Vocabulary:\n{}".format(cv.get_feature_names()))

'''
在 IMDb 电影评论数据上尝试使用 TfidfVectorizer ，并利用网格搜索找出 n 元分词的
最佳设置：
'''
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
# 运行网格搜索需要很长时间，因为网格相对较大，且包含三元分词
param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
"tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)

from sklearn.datasets import load_files
reviews_train = load_files("train/")
# load_files返回一个Bunch对象，其中包含训练文本和训练标签
text_train, y_train = reviews_train.data, reviews_train.target
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
# grid.fit(text_train, y_train)

'''
Best cross-validation score: 0.91
Best parameters:
{'tfidfvectorizer__ngram_range': (1, 3), 'logisticregression__C': 100}
'''

'''
从结果中可以看出，我们添加了二元分词特征与三元分词特征之后，性能提高了一个百分
点多一点。我们可以将交叉验证精度作为 ngram_range 和 C 参数的函数并用热图可视化，
正如我们在第 5 章中所做的那样
'''
# print("Best cross-validation score: {:.2f}".format(grid.best_score_))
# print("Best parameters:\n{}".format(grid.best_params_))
# 从网格搜索中提取分数
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
import mglearn
import matplotlib.pyplot as plt



# 热图可视化
heatmap = mglearn.tools.heatmap(
    scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
    xticklabels=param_grid['logisticregression__C'],
    yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)
plt.show()






























































