# -*- coding: utf-8 -*-

# @Time  : 2019/12/12 13:12

# @Author : Mr.Lin


'''

用于机器学习的文本表示有一种最简单的方法，也是最有效且最常用的方法，就是使用词
袋（bag-of-words）表示。使用这种表示方式时，我们舍弃了输入文本中的大部分结构，如
章节、段落、句子和格式，只计算语料库中每个单词在每个文本中的出现频次。舍弃结构
并仅计算单词出现次数，这会让脑海中出现将文本表示为“袋”的画面。
对于文档语料库，计算词袋表示包括以下三个步骤。
(1) 分词（tokenization）。将每个文档划分为出现在其中的单词 [ 称为词例（token）]，比如
按空格和标点划分。
(2) 构建词表（vocabulary building）。收集一个词表，里面包含出现在任意文档中的所有词，
并对它们进行编号（比如按字母顺序排序）。
(3) 编码（encoding）。对于每个文档，计算词表中每个单词在该文档中的出现频次。
'''

'''
将词袋应用于玩具数据集
词袋表示是在 CountVectorizer 中实现的，它是一个变换器（transformer）。我们首先将它
应用于一个包含两个样本的玩具数据集，来看一下它的工作原理
'''

bards_words =["The fool doth think he is wise,",
"but the wise man knows himself to be a fool"]


'''
导入 CountVectorizer 并将其实例化，然后对玩具数据进行拟合
'''



from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)

'''
拟合 CountVectorizer 包括训练数据的分词与词表的构建，我们可以通过 vocabulary_ 属性
来访问词表：
'''

'''
Vocabulary size: 13
Vocabulary content:
 {'the': 9, 'fool': 3, 'doth': 2, 'think': 10,
  'he': 4, 'is': 6, 'wise': 12, 'but': 1, 'man': 8, 'knows': 7, 
  'himself': 5, 'to': 11, 'be': 0}
'''
# print("Vocabulary size: {}".format(len(vect.vocabulary_)))
# print("Vocabulary content:\n {}".format(vect.vocabulary_))


'''
词表共包含 13 个词，从 "be" 到 "wise" 。
我们可以调用 transform 方法来创建训练数据的词袋表示
'''


'''
bag_of_words: <2x13 sparse matrix of type '<class 'numpy.int64'>'
	with 16 stored elements in Compressed Sparse Row format>
'''

'''
词袋表示保存在一个 SciPy 稀疏矩阵中，这种数据格式只保存非零元素（参见第 1 章）。这
个矩阵的形状为 2×13，每行对应于两个数据点之一，每个特征对应于词表中的一个单词。
这里使用稀疏矩阵，是因为大多数文档都只包含词表中的一小部分单词，也就是说，特征
数组中的大部分元素都为 0。想想看，与所有英语单词（这是词表的建模对象）相比，一
篇电影评论中可能出现多少个不同的单词。保存所有 0 的代价很高，也浪费内存。要想查
看稀疏矩阵的实际内容，可以使用 toarray 方法将其转换为“密集的”NumPy 数组（保存
所有 0 元素）
'''
bag_of_words = vect.transform(bards_words)
# print("bag_of_words: {}".format(repr(bag_of_words)))


'''
Dense representation of bag_of_words:
[[0 0 1 1 1 0 1 0 0 1 1 0 1]
 [1 1 0 1 0 1 0 1 1 1 0 1 1]]
'''
'''
可以看到，每个单词的计数都是 0 或 1。 bards_words 中的两个字符串都没有包含相同
的单词。我们来看一下如何阅读这些特征向量。第一个字符串（ "The fool doth think he
is wise," ）被表示为第一行，对于词表中的第一个单词 "be" ，出现 0 次。对于词表中的第
二个单词 "but" ，出现 0 次。对于词表中的第三个单词 "doth" ，出现 1 次，以此类推。通
过观察这两行可以看出，第 4 个单词 "fool" 、第 10 个单词 "the" 与第 13 个单词 "wise" 同
时出现在两个字符串中。
'''
# print("Dense representation of bag_of_words:\n{}".format(
# bag_of_words.toarray()))

'''
将词袋应用于电影评论
'''


from sklearn.datasets import load_files
reviews_train = load_files("train/")
# load_files返回一个Bunch对象，其中包含训练文本和训练标签
text_train, y_train = reviews_train.data, reviews_train.target
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
'''
X_train:
<25000x74849 sparse matrix of type '<class 'numpy.int64'>'
	with 3445861 stored elements in Compressed Sparse Row format>
'''

'''
X_train 是训练数据的词袋表示，其形状为 25 000×74 849，这表示词表中包含 74 849 个
元素。数据同样被保存为 SciPy 稀疏矩阵。我们来更详细地看一下这个词表。访问词表的
另一种方法是使用向量器（vectorizer）的 get_feature_name 方法，它将返回一个列表，每
个元素对应于一个特征
'''
# print("X_train:\n{}".format(repr(X_train)))

# feature_names = vect.get_feature_names()

'''
Number of features: 74849
First 20 features:
['00', '000', '0000000000001', '00001', '00015', '000s', '001', '003830', '006', '007', '0079', '0080', '0083', '0093638', '00am', '00pm', '00s', '01', '01pm', '02']
Features 20010 to 20030:
['dratted', 'draub', 'draught', 'draughts', 'draughtswoman', 'draw', 'drawback', 'drawbacks', 'drawer', 'drawers', 'drawing', 'drawings', 'drawl', 'drawled', 'drawling', 'drawn', 'draws', 'draza', 'dre', 'drea']
Every 2000th feature:
['00', 'aesir', 'aquarian', 'barking', 'blustering', 'bête', 'chicanery', 'condensing', 'cunning', 'detox', 'draper', 'enshrined', 'favorit', 'freezer', 'goldman', 'hasan', 'huitieme', 'intelligible', 'kantrowitz', 'lawful', 'maars', 'megalunged', 'mostey', 'norrland', 'padilla', 'pincher', 'promisingly', 'receptionist', 'rivals', 'schnaas', 'shunning', 'sparse', 'subset', 'temptations', 'treatises', 'unproven', 'walkman', 'xylophonist']
'''
# print("Number of features: {}".format(len(feature_names)))
# print("First 20 features:\n{}".format(feature_names[:20]))
# print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
# print("Every 2000th feature:\n{}".format(feature_names[::2000]))


'''

如你所见，词表的前 10 个元素都是数字，这可能有些出人意料。所有这些数字都出现在
评论中的某处，因此被提取为单词。大部分数字都没有一目了然的语义，除了 "007" ，在
电影的特定语境中它可能指的是詹姆斯 • 邦德（James Bond）这个角色。 8 从无意义的“单
词”中挑出有意义的有时很困难。进一步观察这个词表，我们发现许多以“dra”开头的英
语单词。你可能注意到了，对于 "draught" 、 "drawback" 和 "drawer" ，其单数和复数形式
都包含在词表中，并且作为不同的单词。这些单词具有密切相关的语义，将它们作为不同
的单词进行计数（对应于不同的特征）可能不太合适。
在尝试改进特征提取之前，我们先通过实际构建一个分类器来得到性能的量化度量。我们
将训练标签保存在 y_train 中，训练数据的词袋表示保存在 X_train 中，因此我们可以在
这个数据上训练一个分类器。对于这样的高维稀疏数据，类似 LogisticRegression 的线性
模型通常效果最好。
我们首先使用交叉验证对 LogisticRegression 进行评估：
'''


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
# scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)

'''

Mean cross-validation accuracy: 0.88

'''
'''
得到的交叉验证平均分数是 88%，这对于平衡的二分类任务来说是一个合理的性能。
我们知道， LogisticRegression 有一个正则化参数 C ，我们可以通过交叉验证来调节它：
'''
# print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))


from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid.fit(X_train, y_train)

'''
Best cross-validation score: 0.89
Best parameters:  {'C': 0.1}
'''
# print("Best cross-validation score: {:.2f}".format(grid.best_score_))
# print("Best parameters: ", grid.best_params_)
'''
我们使用 C=0.1 得到的交叉验证分数是 89%。现在，我们可以在测试集上评估这个参数设
置的泛化性能：
'''

reviews_test = load_files("test/")
text_test, y_test = reviews_test.data, reviews_test.target
X_test = vect.transform(text_test)

'''
0.88

'''
# print("{:.2f}".format(grid.score(X_test, y_test)))

'''
下面我们来看一下能否改进单词提取。 CountVectorizer 使用正则表达式提取词例。默认
使用的正则表达式是 "\b\w\w+\b" 。如果你不熟悉正则表达式，它的含义是找到所有包含
至少两个字母或数字（ \w ）且被词边界（ \b ）分隔的字符序列。它不会匹配只有一个字母
的单词，还会将类似“doesn't”或“bit.ly”之类的缩写分开，但它会将“h8ter”匹配为
一个单词。然后， CountVectorizer 将所有单词转换为小写字母，这样“soon”“Soon”和
“sOon”都对应于同一个词例（因此也对应于同一个特征）。这一简单机制在实践中的效果
很好，但正如前面所见，我们得到了许多不包含信息量的特征（比如数字）。减少这种特
征的一种方法是，仅使用至少在 2 个文档（或者至少 5 个，等等）中出现过的词例。仅在
一个文档中出现的词例不太可能出现在测试集中，因此没什么用。我们可以用 min_df 参数
来设置词例至少需要在多少个文档中出现过：
'''

# vect = CountVectorizer(min_df=5).fit(text_train)
# X_train = vect.transform(text_train)

'''
X_train with min_df: <25000x27272 sparse matrix of type '<class 'numpy.int64'>'
	with 3368680 stored elements in Compressed Sparse Row format>
'''
'''
通过要求每个词例至少在 5 个文档中出现过，我们可以将特征数量减少到 27 271 个，正如
上面的输出所示——只有原始特征的三分之一左右。我们再来查看一些词例：
'''
# print("X_train with min_df: {}".format(repr(X_train)))

feature_names = vect.get_feature_names()
'''
First 50 features:
['00', '000', '0000000000001', '00001', '00015', '000s', '001', '003830', '006', '007', '0079', '0080', '0083', '0093638', '00am', '00pm', '00s', '01', '01pm', '02', '020410', '029', '03', '04', '041', '05', '050', '06', '06th', '07', '08', '087', '089', '08th', '09', '0f', '0ne', '0r', '0s', '10', '100', '1000', '1000000', '10000000000000', '1000lb', '1000s', '1001', '100b', '100k', '100m']
Features 20010 to 20030:
['dratted', 'draub', 'draught', 'draughts', 'draughtswoman', 'draw', 'drawback', 'drawbacks', 'drawer', 'drawers', 'drawing', 'drawings', 'drawl', 'drawled', 'drawling', 'drawn', 'draws', 'draza', 'dre', 'drea']
Every 700th feature:
['00', '40s', 'accent', 'aforementioned', 'aloysius', 'annoucing', 'aristocratic', 'attired', 'bainter', 'bayonets', 'bersen', 'blaze', 'bookstores', 'brighton', 'bursting', 'capers', 'cbbc', 'chemstrand', 'clad', 'colin', 'condensing', 'coolidge', 'cray', 'cusp', 'dears', 'dempster', 'dialing', 'dislocated', 'donnacha', 'duchaussoy', 'eeriest', 'enactment', 'erupt', 'exoskeleton', 'fare', 'figuring', 'flowing', 'fraternal', 'gait', 'gesture', 'goldman', 'grimmer', 'hagan', 'haun', 'heterosexism', 'honhyol', 'hushed', 'import', 'ingenious', 'introspection', 'jardine', 'juli', 'kevetch', 'kovacks', 'larval', 'lev', 'locoformovies', 'luxor', 'managers', 'masterton', 'megalunged', 'mikhalkov', 'modernization', 'mountie', 'naidu', 'newswomen', 'nudeness', 'ominous', 'outwits', 'papers', 'peeping', 'phineas', 'plethora', 'pottery', 'procedure', 'pulpits', 'radulescu', 'reassured', 'reiterates', 'restaraunt', 'rivals', 'ruckus', 'sanguine', 'schygula', 'sensation', 'shelob', 'signpost', 'slipstream', 'solutions', 'splaining', 'steakley', 'stronger', 'superball', 'synch', 'techies', 'thier', 'tokes', 'traped', 'turiquistan', 'undercard', 'unproven', 'validate', 'villan', 'warbeck', 'whiners', 'wonderland', 'yeoh']
'''
# print("First 50 features:\n{}".format(feature_names[:50]))
# print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
# print("Every 700th feature:\n{}".format(feature_names[::700]))

'''
停用词
删除没有信息量的单词还有另一种方法，就是舍弃那些出现次数太多以至于没有信息量的单
词。有两种主要方法：使用特定语言的停用词（stopword）列表，或者舍弃那些出现过于频
繁的单词。 scikit-learn 的 feature_extraction.text 模块中提供了英语停用词的内置列表：
'''
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
'''
Number of stop words: 318
Every 10th stopword:
['part', 'few', 'up', 'sometimes', 'each', 'everything', 'three', 'thus', 'herself', 'hereupon', 'that', 'mostly', 'anywhere', 'nobody', 'or', 'no', 'upon', 'if', 'mill', 'namely', 'beside', 'what', 'go', 'himself', 'amongst', 'why', 'but', 'amount', 'show', 'further', 'enough', 'also']
'''
# print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
# print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

'''
显然，删除上述列表中的停用词只能使特征数量减少 318 个（即上述列表的长度），但可
能会提高性能
'''



'''
另一种方法是按照我们预计的特征信息量大小来缩放特征，而不是舍弃那些认为不重要的
特征。最常见的一种做法就是使用词频 - 逆向文档频率（term frequency–inverse document
frequency，tf-idf）方法。这一方法对在某个特定文档中经常出现的术语给予很高的权
重，但对在语料库的许多文档中都经常出现的术语给予的权重却不高。如果一个单词在
某个特定文档中经常出现，但在许多文档中却不常出现，那么这个单词很可能是对文
档内容的很好描述。 scikit-learn 在两个类中实现了 tf-idf 方法： TfidfTransformer 和
TfidfVectorizer ，前者接受 CountVectorizer 生成的稀疏矩阵并将其变换，后者接受文本
数据并完成词袋特征提取与 tf-idf 变换。tf-idf 缩放方案有几种变体，你可以在维基百科
上阅读相关内容（https://en.wikipedia.org/wiki/Tf-idf）。单词 w 在文档 d 中的 tf-idf 分数在
TfidfTransformer 类和 TfidfVectorizer 类中都有实现
'''

'''
由于 tf-idf 实际上利用了训练数据的统计学属性，所以我们将使用在第 6 章中介绍过的管
道，以确保网格搜索的结果有效。这样会得到下列代码：
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=5),
LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
# grid.fit(text_train, y_train)
'''
Best cross-validation score: 0.89

'''
# print("Best cross-validation score: {:.2f}".format(grid.best_score_))


vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]


# 变换训练数据集
X_train = vectorizer.transform(text_train)
# 找到数据集中每个特征的最大值
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# 获取特征名称
feature_names = np.array(vectorizer.get_feature_names())
print("Features with lowest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf: \n{}".format(feature_names[sorted_by_tfidf[-20:]]))








































