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

def classify(train,train_labels,test,test_labels,method,SVMtype):
    ovoclf = svm.SVC(decision_function_shape=SVMtype)
    ovoclf.fit(train,train_labels)
    print("Accuracy in ",method," = ",ovoclf.score(test,test_labels))

train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

train_pts = train_data.drop('Activity', axis=1)
train_labels = train_data['Activity']

test_pts = test_data.drop('Activity', axis=1)
test_labels = test_data['Activity']



#pca = decomposition.PCA(n_components=40)
#train_pca = pca.fit_transform(train_pts)
#print(pca.explained_variance_ratio_)
#test_pca = pca.fit_transform(test_pts)
#print(pca.explained_variance_ratio_)

#classify(train_pca,train_labels,test_pca,test_labels,"PCA(ovr)",'ovr')
classify(train_pts,train_labels,test_pts,test_labels,"Normal",'ovr')