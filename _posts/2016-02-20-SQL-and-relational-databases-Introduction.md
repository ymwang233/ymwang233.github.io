---
layout: post
title: "SQL及关系型数据库入门"
author: "yphuang"
date: "2016-02-20"
categories: blog
tags: [SQL,MySQL,R]

---



## 什么是关系型数据库及数据库管理系统

数据库(Database)是一种数据的组织和存储方式，通常用于处理结构化的数据。

而关系型数据库(Relational Database)指的是创建在关系模型的基础上的数据库。它借助于集合代数等数学概念和方法来处理数据库中的数据。

数据库管理系统(DBMS,*Database Management System*),是一种专门用于存储、修改并从数据库提取信息的系统软件。

主流的关系型数据库管理系统主要有：MySQL，ORACLE, MS ACCESS，DB2等。

其中，MySQL属于开源软件，而其他的主流数据库管理系统基本都是商业软件。由于很多互联网公司数据库都是采用MySQL进行数据库的管理，所以今天我们主要介绍MySQL的安装、配置及其与R的交互。

## 什么是SQL语言

- SQL(Structured Query Language)是一种专门用来与数据库进行沟通的语言。

- 使用SQL可以对数据库中的数据进行增、删、查、改、权限管理等操作。

- 常用关键词：`SELECT`,`UPDATE`,`DELETE`,`INSERT`.

- 管理关键词：`CREATE`,`ALTER`,`DROP`

- 常用子句、关键词：`FROM`,`WHERE`,`GROUP BY`,`ORDER BY`


## 为什么要学习SQL

很多同学可能会很疑惑：对于数据的增删查改等需求，我们的R不是已经有非常方便的操作了吗？为什么还要多学一门语言呢？

R擅长的领域是数据分析，然而，对于数据存储，却存在很大的问题。一个非常明显的缺陷是：**所有数据均要读入内存**。这就造成了R能处理的数据量存在一个瓶颈。当我们要处理的数据观测数达到上亿级别的时候，R就显得力所不逮了。

数据库能解决的当然不止这一个问题。

当我们遇到如下情景时，数据库就显得非常重要了：

- 当你的数据需要通过网站在线展示；

- 当你在一个团队中工作，你和你的协作成员要同时操作同一个数据集；

- 当你需要为不同的数据用户赋予不同的使用权限；

- 当你要处理的数据量超过了你的电脑内存；

- 当你面对的数据集非常复杂，不能统一组织到一个数据集中时；

- 当你的数据量非常庞大，但你又经常要频繁地获取它的一些子集时；

- 当你的几个数据集关联性很大，更新一个数据集需要同时更新另外一些数据集时；

- 当你对数据的格式要求很严格时。


而如果我们经常与数据打交道，以上的问题是必不可免的。

可见，要想成为一名优秀的数据科学家，学习SQL还是非常有必要的。

当然，SQL虽然是一门语言，但是它有一些非常吸引人的优点：

- 几乎所有重要的DBMS都支持SQL；

- SQL语法简明，简单易学；

- SQL非常灵活，功能强大。


所以，虽然又得多学一门语言，但是也不必苦恼。想想能够几天掌握一门新的语言，也是挺让人激动的呢:)



## MySQL的安装及环境配置

MySQL是一款开软的数据库管理系统，因此我们可以通过在官网进行软件的自由下载安装。

对于入门的同学来说，MySQL Community Server和MySQL Workbench CE结合起来使用是一个不错的开始。MySQL Workbench CE是MySQL的一个开发环境，具有非常友好的交互界面。它跟MySQL的关系如同Rstudio和R的关系。



### 下载地址

- [Download MySQL Installer](http://dev.mysql.com/downloads/installer/)

- [Download MySQL Workbench](http://dev.mysql.com/downloads/workbench/)

### 安装配置

MySQL的安装配置非常简单，一路NEXT就好。如果实在是遇到麻烦，可以用搜索引擎搜索一下安装配置的方法,当然，官网上也有非常详细的安装及使用文档：<http://dev.mysql.com/doc/workbench/en/>.


## SQL基本操作——案例学习

安装完毕，我们就可以启动MySQL Workbench进行数据库的创建等操作了。先使用root用户身份（在安装的过程中创建）进入管理界面。


### 建立一个数据库

新建一个SQL脚本，即可以开始MySQL的编程了。选中某一个代码块，使用`CTRL+ENTER`快捷键即可运行代码。


```
create database db1;

show databases;

-- 创建一个普通用户

CREATE USER yy@localhost IDENTIFIED BY '123';

```


### 建立一个表格

```
use db1;

create table birthdays(
	nameid INTEGER NOT NULL AUTO_INCREMENT,
    firstname varchar(100) not null,
    lastname varchar(100) not null,
    birthday date,
    primary key (nameid)
);

```

### 添加观测数据

```

insert into birthdays(firstname,lastname,birthday)
	values ('peter','Pascal','1991-02-01'),
			('paul','panini','1992-03-02');


```


### 使用查询语句

```
select * from birthdays;

select birthday from birthdays;

```

### 追加数据

```

insert into birthdays(nameid,firstname,lastname,birthday)
	values (10,"Donald","Docker","1934-06-09");
    

```


## SQL与R的交互

R与SQL交互的拓展包非常丰富，不过大致可以分为三大类：

1. 依赖于`DBI`package，如`RMySQL`,`ROracle`,`RPosttgreSQL`,`RSQLite`。这种方式通过与DBMS建立原始的连接实现数据库操作。

2. 依赖于`RODBC`package。这个包通过打开数据库连接驱动的方式建立非直接的连接。如通过依赖于jre读入XLS/XLSX表格的数据。

3. 通过`dplyr` package.


今天主要介绍第1种及第三种方式。


### R连接MySQL

#### 操作数据库中的数据

下面，我们通过R来操作前面在MySQL中建立的数据库`db1`。


```r
library(RMySQL)

# 建立一个连接
mydb <- dbConnect(MySQL(),user="root",
                  password = "mycode",
                  dbname = "db1")

#查看表格
dbListTables(mydb)
```

```
## [1] "birthdays" "mtcars"    "test"
```

```r
#查看某一列
dbListFields(mydb,"birthdays")
```

```
## [1] "nameid"    "firstname" "lastname"  "birthday"
```

```r
#
#dbClearResult(dbListResults(mydb)[[1]])

# 建立一个查询
rs <- dbSendQuery(mydb,"select * from birthdays")
data<-fetch(rs,n = -1)
head(data)
```

```
##   nameid firstname lastname   birthday
## 1      1     peter   Pascal 1991-02-01
## 2      2      paul   panini 1992-03-02
## 3     10    Donald   Docker 1934-06-09
```

```r
# 另一种方法：建立一个查询
dbGetQuery(mydb,"select * from birthdays")
```

```
##   nameid firstname lastname   birthday
## 1      1     peter   Pascal 1991-02-01
## 2      2      paul   panini 1992-03-02
## 3     10    Donald   Docker 1934-06-09
```


#### 将R中的`data.frame`存储到数据库



```r
#将一个data frame对象存储为一个表格
dbWriteTable(mydb,name = "mtcars",value = mtcars,overwrite=TRUE)
```

```
## [1] TRUE
```

```r
#查看结果
dbListTables(mydb)
```

```
## [1] "birthdays" "mtcars"    "test"
```



### 使用dplyr进行数据库操作

`dplyr`是Hadley大神开发的一个专注于data frame类型的数据操作的一个包。它拥有非常简洁、便于记忆、异常丰富的一系列操作函数。更吸引人的是：它支持对sqlite,mysql,postgresql等开源数据库的操作。也就是说：你无需掌握SQL语言也能轻松进行数据库操作。

当然，dplyr并不能替代全部的SQL语言。它主要用于产生分析中最频繁使用的`SELECT`语句。

下面我们看看这是如何做到的。


```r
library(dplyr)

conDplyr<-src_mysql(dbname = "db1",user = "root",password = "mycode",host = "localhost")

mydata<-conDplyr %>% 
  tbl("mtcars") %>%
  select(mpg,cyl,gear) %>%
  filter(gear == 4) %>%
  collect()

head(mydata)
```

```
## Source: local data frame [6 x 3]
## 
##     mpg   cyl  gear
##   (dbl) (dbl) (dbl)
## 1  21.0     6     4
## 2  21.0     6     4
## 3  22.8     4     4
## 4  24.4     4     4
## 5  22.8     4     4
## 6  19.2     6     4
```


### dplyr中的惰性求值

- dplyr只有在必要的情况下才会执行操作

- 它在必要的情况下才会从数据库中载入数据

- 每一个操作函数在执行的时候，并未开始真正从数据库中请求，而是在必要的情况下，一起执行.

如以下的一系列操作并未开始执行数据提取：


```r
library(dplyr)

myDF <- tbl(conDplyr,"mtcars")
myDF1<-filter(myDF,gear == 4)
myDF2<-select(myDF1,mpg,cyl,gear)
```

直到执行以下语句，才真正开始从数据库中提取数据。


```r
head(myDF2)
```

```
##    mpg cyl gear
## 1 21.0   6    4
## 2 21.0   6    4
## 3 22.8   4    4
## 4 24.4   4    4
## 5 22.8   4    4
## 6 19.2   6    4
```


## MySQL深入学习

- 快速入门：『SQL必知必会』。这本书非常简明概要，可以一口气看完。

- 从入门到精通：『MySQL高效编程』。这本书涵盖了非常丰富的学习案例。



## 参考文献

- 『Automated Data Collection with R』第7章

- [Exploring data from database: MySQL, R and dplyr](http://www.unomaha.edu/mahbubulmajumder/data-science/fall-2014/lectures/20-database-mysql/20-database-mysql.html#/)

- [Accessing MySQL through R](http://playingwithr.blogspot.sg/2011/05/accessing-mysql-through-r.html)

