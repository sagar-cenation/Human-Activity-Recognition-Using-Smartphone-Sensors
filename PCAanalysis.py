import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import svm

import matplotlib.pyplot as plt

# This is for the analysis of all the components and their coefficients for the construction
# of dataset. With this we will analyse which components can be removed from 561

train_data = pd.read_csv('datasets/train.csv')

train_pts = train_data.drop('Activity', axis=1)
train_pts = train_pts.drop('subject', axis=1)
train_labels = train_data['Activity']

# pca = PCA(n_components=200)
# train_pca = pca.fit_transform(train_pts, y=train_labels)

# components = pca.components_

# comp = []
# for i in range(0,200):
#     comp.append('comp'+str(i))

# final_comp = pd.DataFrame(components, columns=train_pts.columns.values, index=comp)
# final_comp.to_csv("datasets/compCoeff.csv",sep=",")


# Here we will read the the preserved values of each dataset components



# comp_coeff = pd.read_csv("datasets/compCoeff.csv")
# sums = []
# sumabs = []
# compT = components.T
# s = 0
# for i in range(0,561):
#     for j in range(0,200):
#         s = s + abs(compT[i][j])
#     sumabs.append(s)
#     s = 0
#     sums.append(compT[i].sum())

# x = range(0,561)
# plt.plot(x,sums,label="Sum")
# plt.plot(x,sumabs,label="Absolute sum")
# plt.xlabel("Component No.")
# plt.ylabel("Coefficient sum")
# plt.legend(loc="best")
# plt.show()

# Finding correlation in various features

# correlation = train_pts.corr()
# print(correlation)
# correlation.to_csv("datasets/correlation.csv",sep=",")