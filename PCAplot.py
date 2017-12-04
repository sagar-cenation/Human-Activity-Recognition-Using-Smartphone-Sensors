import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

train_data = pd.read_csv('datasets/train.csv')
train_pts = train_data.drop('Activity', axis=1)
train_labels = train_data['Activity']

comp = []
for i in range(0,73):
    comp.append('comp'+str(i))
pca = PCA(n_components=73)
train_pca = pca.fit_transform(train_pts,y=train_labels)
train_pca = train_pca.tolist()
print(type(train_pca))
final_data = pd.DataFrame(train_pca, index=train_data['Activity'], columns=comp)
final_data.to_csv("datasets/trainPCA.csv",sep=",")