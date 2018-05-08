import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as cf
from sklearn import svm
from sklearn.metrics.pairwise import laplacian_kernel


def classify(train, train_labels, test, test_labels, method):
    """
    train - Training data
    train_labels - Labels corresponding to training data points
    test - Test data
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(train, train_labels)
    print("Predict start = ",time.clock())
    print("Accuracy in ", method, " = ", clf.score(test, test_labels))
    print("Predict end = ",time.clock())
    # print(clf.coef_)
    return clf


def plot_confusion(classifier, test_pts, test_labels):
    classes = ['STANDING',
               'SITTING',
               'LAYING',
               'WALKING',
               'WALKING_DOWNSTAIRS',
               'WALKING_UPSTAIRS']
    pred_label = classifier.predict(test_pts)
    # print(true_label)
    result = cf(test_labels, pred_label, labels=classes)
    res_nor = np.ndarray((6, 6), dtype=float)
    for i in range(0, 6):
        s = result[i].sum()
        for j in range(0, 6):
            res_nor[i][j] = float(result[i][j] / s)
    print(result)
    print(res_nor)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(res_nor)
    # plt.matshow(result)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.legend(loc='best')
    plt.show()


train_data = pd.read_csv('datasets/finaltrain.csv')
test_data = pd.read_csv('datasets/finaltest.csv')

train_pts = train_data.drop('Activity', axis=1)
train_pts = train_pts.drop('subject', axis=1)
train_labels = train_data['Activity']

test_pts = test_data.drop('Activity', axis=1)
test_pts = test_pts.drop('subject', axis=1)
test_labels = test_data['Activity']

# Random scaling the train and test data
# x = 1
# for i in train_pts.columns.values:
#     train_pts.loc[:,i] *= x
#     train_pts.loc[:,i] += (10-x)
#     x = x+1
#     if x == 10:
#         x = 1
# x = 1
# for i in test_pts.columns.values:
#     test_pts.loc[:,i] *= x
#     test_pts.loc[:,i] += (10-x)
#     x = x+1
#     if x == 10:
#         x = 1

# Classifying and plotting after applying PCA
pca = PCA(n_components=200)
train_pca = pca.fit_transform(train_pts, y=train_labels)
# print(pca.explained_variance_ratio_.sum())
print("Transform start = ",time.clock())
test_pca = pca.transform(test_pts)
print("Transform end = ",time.clock())
clf = classify(train_data, train_labels, test_data, test_labels, "PCA(RBF)")
# y_pred = clf.predict(test_pca)
# target_names = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS',
#                 'WALKING_UPSTAIRS']
# result = cf(test_labels, y_pred, labels=target_names)

# print(classification_report(y_pred, test_labels, target_names=target_names))
# print(result)
# plot_confusion(clf, test_pca, test_labels)

# Plotting PCA components vs accuracy graph
# accuTrain = []
# accuTest = []
# for i in range(1,100):
#     pca = PCA(n_components=i)
#     train_pca = pca.fit_transform(train_pts,y=train_labels)
#     # print(pca.explained_variance_ratio_.sum())
#     test_pca = pca.transform(test_pts)
#     clf = svm.LinearSVC()
#     clf.fit(train_pca,train_labels)
#     accuTest.append(clf.score(test_pca,test_labels))
#     accuTrain.append(clf.score(train_pca,train_labels))
# comp = [i for i in range(1,100)]
# plt.plot(comp,accuTrain,label="Train Accuracy")
# plt.plot(comp,accuTest,label="Test Accuracy")
# plt.xlabel("No. of Components")
# plt.ylabel("Accuracy")
# plt.title("Accuracy V/S Components")
# plt.legend(loc='best')
# plt.show()

# Classifying and plotting Actual dataset
#ovoclf = classify(train_pts, train_labels, test_pts, test_labels, "Normal", 'ovr')
#plot_confusion(ovoclf, test_pts, test_labels)
