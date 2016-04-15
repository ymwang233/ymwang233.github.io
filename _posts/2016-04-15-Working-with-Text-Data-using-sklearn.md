---
layout: post
title: "使用scikit-learn进行文本分类"
author: "yphuang"
date: "2016-03-21"
categories: blog
tags: [Python,文本分类,scikit-learn]

---



# 使用scikit-learn进行文本分类

## scikit-learn简介

[scikit-learn](http://scikit-learn.org/stable/)是Python最为流行的一个机器学习库。它具有如下吸引人的特点：

- 简单、高效且异常丰富的数据挖掘/数据分析算法实现；
- 基于NumPy,SciPy,以及matplotlib，从数据探索性分析，数据可视化到算法实现，整个过程一体化实现；
- 开源，有非常丰富的学习文档。

尤其是当我们要进行多种算法的效果对比评价，这种一体化实现的优势就更加能够凸显出来了。

既然`scikit-learn`模块如此重要，废话不多说，下面马上开搞！

## 项目组织及文件加载

### 项目组织

工作路径：`D:\my_python_workfile\Thesis\sklearn_exercise`

|--data：用于存放数据

    |--20news-bydate：练习用数据集
                |--20news-bydate-train：训练集
                |--20news-bydate-test：测试集
                
                
### 文件加载

假设我们需要加载的数据，组织结构如下：

```
container_folder/
    category_1_folder/
        file_1.txt file_2.txt ... file_42.txt
    category_2_folder/
        file_43.txt file_44.txt ...

```
可以使用以下函数进行数据的加载：

```
sklearn.datasets.load_files(container_path, description=None, categories=None, load_content=True, shuffle=True, encoding=None, decode_error='strict', random_state=0)

```
- 参数解释：
    + `container_path`:container_folder的路径；
    + `load_content = True`:是否把文件中的内容加载到内存；
    + `encoding = None`:编码方式。当前文本文件的编码方式一般为“utf-8”，如果不指明编码方式（encoding=None），那么文件内容将会按照bytes处理，而不是unicode处理。

- 返回值：Bunch Dictionary-like object.主要属性有
    + data:原始数据；
    + filenames:每个文件的名字；
    + target:类别标签（从0开始的整数索引）；
    + target_names:类别标签的具体含义（由子文件夹的名字`category_1_folder`等决定）。
    
下面，即采用这种方式，使用测试数据集[The 20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/)进行实例演示。先从网上下载该数据集，再在本地进行数据的加载。


```python
# 加载库
import os
import sys

##配置utf-8输出环境
#reload(sys)
#sys.setdefaultencoding("utf-8")

# 设置当前工作路径
os.chdir("D:\\my_python_workfile\\Thesis\\sklearn_exercise")

# 加载数据
from sklearn import datasets
twenty_train = datasets.load_files("data/20news-bydate/20news-bydate-train")
twenty_test = datasets.load_files("data/20news-bydate/20news-bydate-test")
```


```python
len(twenty_train.target_names),len(twenty_train.data),len(twenty_train.filenames),len(twenty_test.data)
```




    (20, 11314, 11314, 7532)




```python
print("\n".join(twenty_train.data[0].split("\n")[:3]))
```

    From: cubbie@garnet.berkeley.edu (                               )
    Subject: Re: Cubs behind Marlins? How?
    Article-I.D.: agate.1pt592$f9a
    


```python
print(twenty_train.target_names[twenty_train.target[0]])
```

    rec.sport.baseball
    


```python
twenty_train.target[:10]
```




    array([ 9,  4, 11,  4,  0,  4,  5,  5, 13, 12])



可见，文件已经被成功载入。

当然，作为入门的训练，我们也可以使用`scikit-learn`自带的`toy example`数据集进行测试、玩耍。下面，介绍一下如何加载自带的数据集。



```python
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)
```

## 文本特征提取

文本数据属于非结构化的数据，一般要转换成结构化的数据，方能进行实施机器学习算法实现文本分类。

常见的做法是将文本转换成『文档-词项矩阵』。矩阵中的元素，可以使用词频，或者[TF-IDF值](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)等。

### 计算词频


```python
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words="english",decode_error='ignore')
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
```




    (11314, 129783)



### 使用TF-IDF进行特征提取


```python
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
```




    (11314, 129783)



以上程序使用了两步进行文本的形式化表示：先用`fit()`方法使得模型适用数据；再用`transform()`方法把词频矩阵重新表述成TF-IDF.

如下所示，也可以一步到位进行设置。


```python
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```




    (11314, 129783)




```python

```

## 分类器训练


```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)
```


```python
# 对新的样本进行预测
docs_new = ['God is love','OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc,category in zip(docs_new,predicted):
    print("%r => %s") %(doc,twenty_train.target_names[category])

```

    'God is love' => soc.religion.christian
    'OpenGL on the GPU is fast' => comp.graphics
    

## 分类效果评价

### 建立管道


```python
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect',CountVectorizer(stop_words="english",decode_error='ignore')),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultinomialNB()),
                    ])

text_clf = text_clf.fit(twenty_train.data,twenty_train.target)
```

### 测试集分类准确率


```python
import numpy as np
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
```




    0.81691449814126393



使用朴素贝叶斯分类器，得到的测试集分类准确率为81.7%，效果还不错！

下面，使用线性核支持向量机看看效果如何。


```python
from sklearn.linear_model import SGDClassifier
text_clf_2 = Pipeline([('vect',CountVectorizer(stop_words='english',decode_error='ignore')),
                      ('tfidf',TfidfTransformer()),
                      ('clf',SGDClassifier(loss = 'hinge',penalty = 'l2',
                                          alpha = 1e-3,n_iter = 5, random_state = 42)),
                      ])

_ = text_clf_2.fit(twenty_train.data,twenty_train.target)
predicted = text_clf_2.predict(docs_test)

np.mean(predicted == twenty_test.target)
```




    0.82355284121083383



支持向量机的分类准确率有所提升。

`scikit-learn`中提供了更精细化的评价指标，如：各类别的精确度，召回率，F值等。

下面，我们来看看更详细的指标表现如何。


```python
from sklearn import metrics
print(metrics.classification_report(twenty_test.target,predicted,
                                   target_names = twenty_test.target_names))
```

                              precision    recall  f1-score   support
    
                 alt.atheism       0.71      0.71      0.71       319
               comp.graphics       0.81      0.69      0.74       389
     comp.os.ms-windows.misc       0.72      0.79      0.75       394
    comp.sys.ibm.pc.hardware       0.73      0.66      0.69       392
       comp.sys.mac.hardware       0.82      0.83      0.82       385
              comp.windows.x       0.86      0.77      0.81       395
                misc.forsale       0.80      0.87      0.84       390
                   rec.autos       0.91      0.90      0.90       396
             rec.motorcycles       0.93      0.97      0.95       398
          rec.sport.baseball       0.88      0.91      0.90       397
            rec.sport.hockey       0.87      0.98      0.92       399
                   sci.crypt       0.85      0.96      0.90       396
             sci.electronics       0.80      0.62      0.70       393
                     sci.med       0.90      0.87      0.88       396
                   sci.space       0.84      0.96      0.90       394
      soc.religion.christian       0.75      0.93      0.83       398
          talk.politics.guns       0.70      0.93      0.80       364
       talk.politics.mideast       0.92      0.92      0.92       376
          talk.politics.misc       0.89      0.56      0.69       310
          talk.religion.misc       0.81      0.39      0.53       251
    
                 avg / total       0.83      0.82      0.82      7532
    
    

测试集的精确度和召回率的表现均不错。

下面看看『混淆矩阵』的结果。


```python
metrics.confusion_matrix(twenty_test.target,predicted)
```

## 使用网格搜索进行参数优化

我们使用分类器进行文本分类的过程中，有些参数需要预先给定。如前面`TfidfTransformer()`中的`use_idf`;`MultinomialNB()`中的平滑参数`alpha`;`SGClassifier()`中的惩罚系数`alpha`。然而，参数设置为多少,并不能直接拍脑袋决定。因为参数的设置可能会导致结果天差地别。

为了不沦落为一个『调参狗』，我们来看看如何使用暴力的『网格搜索算法』让计算机帮我们进行参数寻优。


```python
from sklearn.grid_search import GridSearchCV
parameters = {
            'vect__ngram_range':[(1,1),(1,2)],
             'tfidf__use_idf':(True,False),
             'clf__alpha':(1e-2,1e-3)
             }
```

如果要穷尽所有参数的组合，那势必要花费很多时间来等待结果。有的『土豪』同学可能会想：我能不能用金钱来换时间？

答案是肯定的。如果你有一台8核的电脑，那就把所有的核都用上吧！



```python
gs_clf = GridSearchCV(text_clf_2,parameters,n_jobs = -1)
```


```python
gs_clf = gs_clf.fit(twenty_train.data,twenty_train.target)
```

设置`n_jobs = -1`，计算机就会帮你自动检测并用上你所有的核进行并行计算。


```python
best_parameters,score,_ = max(gs_clf.grid_scores_,key = lambda x:x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" %(param_name,best_parameters[param_name]))
```

    clf__alpha: 0.01
    tfidf__use_idf: True
    vect__ngram_range: (1, 1)
    


```python
score
```




    0.90516174650875025



## 参考文献

- [scikit-learn官网](http://scikit-learn.org/stable/)
- [ scikit-learn：0.2. 加载自己的原始数据](http://blog.csdn.net/mmc2015/article/details/46852755)
- <http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html>

- [Working With Text Data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

