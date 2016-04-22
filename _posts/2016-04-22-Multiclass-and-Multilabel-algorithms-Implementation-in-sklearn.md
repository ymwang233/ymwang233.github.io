---
layout: post
title: "使用scikit-learn实现多类别及多标签分类算法"
author: "yphuang"
date: "2016-04-22"
categories: blog
tags: [Python,多标签分类,scikit-learn]

---
# 使用scikit-learn实现多类别及多标签分类算法

## 多标签分类格式

对于多标签分类问题而言，一个样本可能同时属于多个类别。如一个新闻属于多个话题。这种情况下，因变量$$y$$需要使用一个矩阵表达出来。

而多类别分类指的是y的可能取值大于2，但是y所属类别是唯一的。它与多标签分类问题是有严格区别的。所有的scikit-learn分类器都是默认支持多类别分类的。但是，当你需要自己修改算法的时候，也是可以使用`scikit-learn`实现多类别分类的前期数据准备的。

多类别或多标签分类问题，有两种构建分类器的策略：**One-vs-All**及**One-vs-One**。下面，通过一些例子进行演示如何实现这两类策略。



```python
#
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2,3,4],[2],[0,1,3],[0,1,2,3,4],[0,1,2]]
MultiLabelBinarizer().fit_transform(y)

```




    array([[0, 0, 1, 1, 1],
           [0, 0, 1, 0, 0],
           [1, 1, 0, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 0, 0]])



## One-Vs-The-Rest策略

这个策略同时也称为**One-vs-all**策略，即通过构造K个判别式（K为类别的个数），第$$i$$个判别式将样本归为第$$i$$个类别或非第$$i$$个类别。这种分类方法虽然比较耗时间，但是能够通过每个类别对应的判别式获得关于该类别的直观理解（如文本分类中每个话题可以通过只属于该类别的高频特征词区分）。



### 多类别分类学习


```python
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X,y = iris.data,iris.target
OneVsRestClassifier(LinearSVC(random_state = 0)).fit(X,y).predict(X)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])



### 多标签分类学习

Kaggle上有一个关于多标签分类问题的竞赛：[Multi-label classification of printed media articles to topics](https://www.kaggle.com/c/wise-2014)。


关于该竞赛的介绍如下：

> This is a multi-label classification competition for articles coming from Greek printed media. Raw data comes from the scanning of print media, article segmentation, and optical character segmentation, and therefore is quite noisy. Each article is examined by a human annotator and categorized to one or more of the topics being monitored. Topics range from specific persons, products, and companies that can be easily categorized based on keywords, to more general semantic concepts, such as environment or economy. Building multi-label classifiers for the automated annotation of articles into topics can support the work of human annotators by suggesting a list of all topics by order of relevance, or even automate the annotation process for media and/or categories that are easier to predict. This saves valuable time and allows a media monitoring company to expand the portfolio of media being monitored.  


我们从该网站下载[相应的数据](https://www.kaggle.com/c/wise-2014/data)，作为多标签分类的案例学习。

#### 数据描述

这个文本数据集已经用词袋模型进行形式化表示，共201561个特征词，每个文本对应一个或多个标签，共203个分类标签。该网站提供了两种数据格式：`ARFF`和`LIBSVM`,`ARFF`格式的数据主要适用于weka，而`LIBSVM`格式适用于matlab中的`LIBSVM`模块。这里，我们采用`LIBSVM`格式的数据。

数据的每一行以逗号分隔的整数序列开头，代表类别标签。紧接着是以\t分隔的`id:value`对。其中，`id`为特征词的ID，`value`为特征词在该文档中的`TF-IDF`值。

形式如下。

```
58,152 833:0.032582 1123:0.003157 1629:0.038548 ...

```

#### 数据载入


```python
# load modules
import os 
import sys

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
# set working directory
os.chdir("D:\\my_python_workfile\\Thesis\\kaggle_multilabel_classification")

```


```python
# read files
X_train,y_train = load_svmlight_file("./data/wise2014-train.libsvm",dtype=np.float64,multilabel=True)
X_test,y_test = load_svmlight_file("./data/wise2014-test.libsvm",dtype = np.float64,multilabel=True)
```

#### 模型拟合及预测


```python
# transform y into a matrix
mb = MultiLabelBinarizer()
y_train = mb.fit_transform(y_train)

# fit the model and predict

clf = OneVsRestClassifier(LogisticRegression(),n_jobs=-1)
clf.fit(X_train,y_train)
pred_y = clf.predict(X_test)
```

#### 模型评估

由于没有关于测试集的真实标签，这里看看训练集的预测情况。


```python
# training set result
y_predicted = clf.predict(X_train)

#report 
#print(metrics.classification_report(y_train,y_predicted))

import numpy as np
np.mean(y_predicted == y_train)
```




    0.99604661023482433



#### 保存结果


```python
# write the output
out_file = open("pred.csv","w")
out_file.write("ArticleId,Labels\n")
id = 64858

for i in xrange(pred_y.shape[0]):
    label = list(mb.classes_[np.where(pred_y[i,:]==1)[0]].astype("int"))
    label = " ".join(map(str,label))
    if label == "":  # if the label is empty
        label = "103"
    out_file.write(str(id+i)+","+label+"\n")
out_file.close()
```

## One-Vs-One策略

**One-Vs-One**策略即是两两类别之间建立一个判别式，这样，总共需要$$K(K-1)/2$$个判别式，最后通过投票的方式确定样本所属类别。

### 多类别分类学习



```python
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X,y = iris.data,iris.target
OneVsOneClassifier(LinearSVC(random_state = 0)).fit(X,y).predict(X)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])



## 参考文献

- [Multiclass and multilabel algorithms](http://scikit-learn.org/stable/modules/multiclass.html#multiclass)

- [Greek Media Monitoring Multilabel Classification (WISE 2014)](https://www.kaggle.com/c/wise-2014/forums)
