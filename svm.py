import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb
from pylab import rcParams

import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import confusion_matrix as cf
from sklearn import svm

train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

x_train = train_data.drop('Activity', axis=1)
pca = decomposition.PCA(n_components=40)
x_train_pca = pca.fit_transform(x_train)

print(pca.explained_variance_ratio_)