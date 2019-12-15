# -*- coding: utf-8 -*-

# @Time  : 2019/12/12 14:49

# @Author : Mr.Lin

'''

 CountVectorizer 和 TfidfVectorizer 中的特征提取相对简单，还有更为复杂
的方法。在更加复杂的文本处理应用中，通常需要改进的步骤是词袋模型的第一步：分词
（tokenization）。这一步骤为特征提取定义了一个单词是如何构成的。
我们前面看到，词表中通常同时包含某些单词的单数形式和复数形式，比如 "drawback"
和 "drawbacks" 、 "drawer" 和 "drawers" 、 "drawing" 和 "drawings" 。对于词袋模型而言，
"drawback" 和 "drawbacks" 的语义非常接近，区分二者只会增加过拟合，并导致模型无法
充分利用训练数据。同样我们还发现，词表中包含像 " replace" 、 "replaced" 、 "replace
ment" 、 "replaces" 和 "replacing" 这样的单词，它们都是动词“to replace”的不同动词形
式或相关名词。与名词的单复数形式一样，将不同的动词形式及相关单词视为不同的词
例，这不利于构建具有良好泛化性能的模型

'''


'''
这个问题可以通过用词干（word stem）表示每个单词来解决，这一方法涉及找出 [ 或合并
（conflate）] 所有具有相同词干的单词。如果使用基于规则的启发法来实现（比如删除常见
的后缀），那么通常将其称为词干提取（stemming）。如果使用的是由已知单词形式组成的
字典（明确的且经过人工验证的系统），并且考虑了单词在句子中的作用，那么这个过程
被称为词形还原（lemmatization），单词的标准化形式被称为词元（lemma）。词干提取和
词形还原这两种处理方法都是标准化（normalization）的形式之一，标准化是指尝试提取
一个单词的某种标准形式。标准化的另一个有趣的例子是拼写校正，这种方法在实践中很
有用，但超出了本书的范围。
为了更好地理解标准化，我们来对比一种词干提取方法（Porter 词干提取器，一种广泛使
用的启发法集合，从 nltk 包导入）与 spacy 包
11
中实现的词形还原： 

'''


import spacy

import nltk


# 加载spacy的英语模型
en_nlp = spacy.load('en')
# 将nltk的Porter词干提取器实例化
stemmer = nltk.stem.PorterStemmer()
# 定义一个函数来对比spacy中的词形还原与nltk中的词干提取
def compare_normalization(doc):
    # 在spacy中对文档进行分词
    doc_spacy = en_nlp(doc)
    # 打印出spacy找到的词元
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # 打印出Porter词干提取器找到的词例
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])


'''
将用一个句子来比较词形还原与 Porter 词干提取器，以显示二者的一些区别：
'''
compare_normalization(u"Our meeting today was worse than yesterday, "
"I'm scared of meeting the clients tomorrow.")







































































