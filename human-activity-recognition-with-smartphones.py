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
from sklearn.ensemble import RandomForestClassifier


# Loading the training and testing data sets
train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

# print test_data.head(2)

# Partitioning data to input and target variable

x_train = train_data.drop('Activity', axis=1)

y_train = pd.get_dummies(train_data.Activity)  # create dummies from a categorical column of a dataframe

x_test = test_data.drop('Activity', axis=1)
y_test = pd.get_dummies(test_data.Activity)

# find principal components
pca = decomposition.PCA(n_components=40)
x_train_pca = pca.fit_transform(x_train)
print(pca.explained_variance_ratio_)  # how much info is compressed into the first few components

print(pca.explained_variance_ratio_.sum())  # cumulative variance(figure out how many components to keep...atleast 70% keep)
# value = 1 means 100  of dataset's info is captured the components shown that were returned(we dont want that as it contain noise, redundancy and outliers)

feature_names = x_train.head(0)
x_train = pca.fit(x_train).transform(x_train)

comps = pd.DataFrame(pca.components_)

sb.heatmap(comps)
plt.show()


# finding if any missing values
missing = x_train.isnull().sum()
missing = missing[missing > 0]
print(missing)

# Train Using Random Forest with 20 Trees

# rf = RandomForestClassifier(20)
# rf.fit(x_train, y_train)

# y_pred = rf.predict(x_test)

# cf(np.argmax(y_test.as_matrix(), axis=1), np.argmax(y_pred, axis=1))

# print rf.score(x_test, y_test)

# plt.matshow(cf(np.argmax(y_test.as_matrix(), axis=1), np.argmax(y_pred, axis=1)), cmap='Reds')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# visualizing the data

# groups = train_data.groupby('Activity')

# for name, group in groups:
#     plt.plot(group['tBodyAcc-mean()-Y'], group['tBodyAccMag-mean()'], '.', label=name)

# plt.legend(loc='best')
# plt.show()
