# Name: Zheng Hao Tan
# Email: tanzhao@umich.edu
# Date: February 14, 2016

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

df.tail()

import matplotlib.pyplot as plt

import numpy as np

y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color:'red', marker='o', label='setosa')


plt.scatter(X[:50, 0], X[:50, 1], color:'red', marker='o', label='setosa')
