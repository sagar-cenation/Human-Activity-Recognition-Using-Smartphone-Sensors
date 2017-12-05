import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot import plotGraph, plotGraph3D
from sklearn.decomposition import PCA, KernelPCA

train_data = pd.read_csv('datasets/train.csv')
train_pts = train_data.drop('Activity', axis=1)
train_labels = train_data['Activity']

# test_data = pd.read_csv('datasets/test.csv')
# test_pts = test_data.drop('Activity', axis=1)
# test_labels = test_data['Activity']

pca = KernelPCA(n_components=100)
train_pca = pca.fit(train_pts,train_labels)
y = train_pca.lambdas_
x = range(1,101)
plt.plot(x,y)
plt.xlabel("No. of components")
plt.ylabel("Eigen values")
plt.title("Data preserved w.r.t no. of components")
plt.show()

# comp = []
# for i in range(0,100):
#     comp.append('comp'+str(i))
# pca = KernelPCA(n_components=100, kernel='rbf', gamma=0.1)
# train_pca = pca.fit_transform(train_pts,y=train_labels)
# train_pca = train_pca.tolist()
# print(type(train_pca))
# final_data = pd.DataFrame(train_pca, index=train_data['Activity'], columns=comp)
# final_data.to_csv("datasets/train20KPCARBFG0_1.csv",sep=",")
# test_pca = pca.transform(test_pts)
# test_pca = test_pca.tolist()
# final_test = pd.DataFrame(test_pca, index=test_data['Activity'], columns=comp)
# final_test.to_csv("datasets/test20KPCARBFG0_1.csv", sep=",")

# train_pca = pd.read_csv('datasets/train200PCA.csv')
# plotGraph3D(train_pca,'comp0','comp1','comp4')
