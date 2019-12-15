# -*- coding: utf-8 -*-

# @Time  : 2019/12/12 15:42

# @Author : Mr.Lin



'''

主题建模与文档聚类
常用于文本数据的一种特殊技术是主题建模（topic modeling），这是描述将每个文档分配
给一个或多个主题的任务（通常是无监督的）的概括性术语。这方面一个很好的例子是新
闻数据，它们可以被分为“政治”“体育”“金融”等主题。如果为每个文档分配一个主
题，那么这是一个文档聚类任务，正如第 3 章中所述。如果每个文档可以有多个主题，那
么这个任务与第 3 章中的分解方法有关。我们学到的每个成分对应于一个主题，文档表示
中的成分系数告诉我们这个文档与该主题的相关性强弱。通常来说，人们在谈论主题建模
时，他们指的是一种叫作隐含狄利克雷分布（Latent Dirichlet Allocation，LDA）的特定分
解方法。
'''
from sklearn.feature_extraction.text import CountVectorizer

'''
隐含狄利克雷分布
从直观上来看，LDA 模型试图找出频繁共同出现的单词群组（即主题）。LDA 还要求，每
个文档可以被理解为主题子集的“混合”。重要的是要理解，机器学习模型所谓的“主
题”可能不是我们通常在日常对话中所说的主题，而是更类似于 PCA 或 NMF（第 3 章
讨论过这些内容）所提取的成分，它可能具有语义，也可能没有。即使 LDA“主题”具
有语义，它可能也不是我们通常所说的主题。回到新闻文章的例子，我们可能有许多关
于体育、政治和金融的文章，由两位作者所写。在一篇政治文章中，我们预计可能会看
到“州长”“投票”“党派”等词语，而在一篇体育文章中，我们预计可能会看到类似“队
伍”“得分”和“赛季”之类的词语。这两组词语可能会同时出现，而例如“队伍”和
“州长”就不太可能同时出现。但是，这并不是我们预计可能同时出现的唯一的单词群组。
这两位记者可能偏爱不同的短语或者选择不同的单词。可能其中一人喜欢使用“划界”
（demarcate）这个词，而另一人喜欢使用“两极分化”（polarize）这个词。其他“主题”可
能是“记者 A 常用的词语”和“记者 B 常用的词语”，虽然这并不是通常意义上的主题。
我们将 LDA 应用于电影评论数据集，来看一下它在实践中的效果。对于无监督的文本文档
模型，通常最好删除非常常见的单词，否则它们可能会支配分析过程。我们将删除至少在
15% 的文档中出现过的单词，并在删除前 15% 之后，将词袋模型限定为最常见的 10 000 个
单词：
'''



from sklearn.datasets import load_files
reviews_train = load_files("train/")
# load_files返回一个Bunch对象，其中包含训练文本和训练标签
text_train, y_train = reviews_train.data, reviews_train.target
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)

vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(text_train)
'''
将学习一个包含 10 个主题的主题模型，它包含的主题个数很少，我们可以查看所有
主题。与 NMF 中的分量类似，主题没有内在的顺序，而改变主题数量将会改变所有主
题。 15 我们将使用 "batch" 学习方法，它比默认方法（ "online" ）稍慢，但通常会给出更好
的结果。我们还将增大 max_iter ，这样会得到更好的模型：
'''
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=10, learning_method="batch",
max_iter=25, random_state=0)
# 我们在一个步骤中构建模型并变换数据
# 计算变换需要花点时间，二者同时进行可以节省时间
document_topics = lda.fit_transform(X)


'''
LatentDirichletAllocation 有一个 components_ 属性，
其中保存了每个单词对每个主题的重要性。 components_ 的大小为 (n_topics, n_words) ：
'''
print(lda.components_.shape)

import numpy as np
# 对于每个主题（components_的一行），将特征排序（升序）
# 用[:, ::-1]将行反转，使排序变为降序
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# 从向量器中获取特征名称
feature_names = np.array(vect.get_feature_names())


import mglearn

# 打印出前10个主题：
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
sorting=sorting, topics_per_chunk=5, n_words=10)





























































































