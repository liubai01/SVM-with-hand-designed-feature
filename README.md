# SVM-with-hand-designed-feature
![introduction](https://github.com/liubai01/SVM-with-hand-designed-feature/blob/master/img/f1_modified.png)

An interactive case study/tutorial of [SVM](https://en.wikipedia.org/wiki/Support_vector_machine)(support vector machine) with hand-designed feature in Chinese. The case study bases on MNIST, which aims at building up a binary classifer between '0' and '8'. It is also released on my [blog(Chinese)](https://blog.csdn.net/liubai01/article/details/82119462). The english version is on schedule and will be released soon. This tutorial will help you get into the process of feature design in a classical machine learning process.

### Prerequisite

1. Python3(Basic Concept of function, import, OOP, etc.)
2. [Scikit-Learn](http://scikit-learn.org/stable/documentation.html)(we only use a tiny part of it, don't worry about it!)
3. [Numpy](http://www.numpy.org/)(know some basic operators)
4. Basic Knowledge of SVM(support vector machine)
5. [Jupyter Notebook](http://jupyter.org/)(An interesting tool! An interactive programming environment)

Recommendation:  [Anaconda](https://www.anaconda.com/download/) is a one-stage solution instead of manually installing these 3-rd party package.

### Download and setup (In Ubuntu16 LTS)

1. Open terminal at target directory(anywhere you want to start this interactive tutorial)
2. follow these steps:

```shell
git clone https://github.com/liubai01/SVM-with-hand-designed-feature.git
cd SVM-with-hand-designed-feature/
jupyter notebook
```

Finally, open your browser to connect your Jupyter session. Open `case_study_01.ipynb` directly.

### Goal

1. get a basic command of the usage of numpy
2. learn how to train a SVM classifier base on scikit-learn
3. have fun!

### Acknowledgement

Some figures is quoted from [deep learning book](http://www.deeplearningbook.org/). The model bases on the dataset [MNIST](http://yann.lecun.com/exdb/mnist/).