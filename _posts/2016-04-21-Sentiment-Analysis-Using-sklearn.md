---
layout: post
title: "使用scikit-learn进行电影评论情感分类"
author: "yphuang"
date: "2016-04-21"
categories: blog
tags: [Python,情感分类,scikit-learn]

---



# 使用scikit-learn进行电影评论情感分类

## 数据准备

从网站[Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)下载语料。这里选择`polarity dataset v2.0`。该数据集包含正负情感极性（`pos`和`neg`）的电影评论各1000条。

下面，进行数据载入，并进行训练集/测试集划分。



```python
# load library
import os 
import sys

# set working directory
os.chdir("D:\\my_python_workfile\\Thesis\\movie_review\\review_polarity\\txt_sentoken")

dataset_dir_name = os.getcwd()
dataset_dir_name
```




    'D:\\my_python_workfile\\Thesis\\movie_review\\review_polarity\\txt_sentoken'




```python
# load library
import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# load data,and split into training/test set
movie_reviews = load_files(dataset_dir_name)
 # split 
doc_terms_train,doc_terms_test,doc_class_train,doc_class_test =  train_test_split(
        movie_reviews.data,movie_reviews.target,test_size = 0.2,random_state = None)
    

```


```python
len(doc_class_train),len(doc_class_test),(movie_reviews.target_names)

#print("\n".join(movie_reviews.data[0].split("\n"))[:20])
```




    (1600, 400, ['neg', 'pos'])



## 建立vectorizer/classifier pipeline


```python
# build a vectorizer/classifier pipeline
pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])
```

## 参数寻优


```python
# grid search
parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
    }
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
grid_search.fit(doc_terms_train,doc_class_train)

print(grid_search.grid_scores_)
```

    [mean: 0.83750, std: 0.01659, params: {'vect__ngram_range': (1, 1)}, mean: 0.85938, std: 0.01338, params: {'vect__ngram_range': (1, 2)}]
    

## 模型预测效果评估


```python
# y_predicted
y_predicted = grid_search.predict(doc_terms_test)

# report
print(metrics.classification_report(doc_class_test,y_predicted,
                                   target_names = movie_reviews.target_names))
```

                 precision    recall  f1-score   support
    
            neg       0.85      0.85      0.85       188
            pos       0.87      0.86      0.87       212
    
    avg / total       0.86      0.86      0.86       400
    
    


```python
# confusion matrix
confusion_matrix = metrics.confusion_matrix(doc_class_test,y_predicted)
print(confusion_matrix)
```

    [[160  28]
     [ 29 183]]
    

## 深入学习

以上作为一个入门级的介绍，就到此为止啦～

当然，在现实生活中，我们不能仅仅满足于对电影评论的正负面分类，而应该考虑更细粒度的分类问题。比如电影评论文本分为1~5星，1星和2星之间比1星和5星更为相似，所以这种多分类问题可以看做是ordinal regression问题求解(见参考文献Pang B等)。


正好**kaggle**上有一个更细粒度的情感分类问题:[Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)。对情感分析感兴趣的同学，可以捋起袖子，来一场Kaggle的比赛了。




## 参考文献

- [scikit-learn exercise_02_sentiment.py](https://github.com/scikit-learn/scikit-learn/blob/master/doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py)
- [斯坦福大学自然语言处理第七课“情感分析（Sentiment Analysis）”](http://52opencourse.com/235/%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E7%AC%AC%E4%B8%83%E8%AF%BE-%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90%EF%BC%88sentiment-analysis%EF%BC%89)

- Pang B, Lee L. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales[C]//Proceedings of the 43rd Annual Meeting on Association for Computational Linguistics. Association for Computational Linguistics, 2005: 115-124.


