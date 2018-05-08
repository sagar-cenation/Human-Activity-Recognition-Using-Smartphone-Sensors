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
    clf = svm.SVC(kernel='linear',verbose=True)
    print("SVM train start: ",time.clock())
    clf.fit(train, train_labels)
    print("SVM train stop: ",time.clock())
    print("Accuracy in ", method, " = ", clf.score(test, test_labels))
    print("SVM test stop: ",time.clock())
    # print(clf.coef_)
    return clf


def plot_confusion(classifier, test_pts, test_labels):
    classes = ['STANDING',
               'SITTING',
               'LYING',
               'WALKING',
               'WALKING_DOWNSTAIRS',
               'WALKING_UPSTAIRS']
    cl = ['STANDING',
               'SITTING',
               'LYING',
               'WALK',
               'WALK_DOWN',
               'WALK_UP']
    pred_label = classifier.predict(test_pts)
    # print(true_label)
    result = cf(test_labels, pred_label, labels=classes)
    # res_nor = np.ndarray((6, 6), dtype=float)
    # for i in range(0, 6):
    #     s = result[i].sum()
    #     for j in range(0, 6):
    #         res_nor[i][j] = float(result[i][j] / s)
    print(result)
    # print(res_nor)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(result)
    # plt.matshow(result)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + cl)
    ax.set_yticklabels([''] + cl)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.legend(loc='best')
    plt.show()


train_data = pd.read_csv('datasets/finaltrain.csv')
test_data = pd.read_csv('datasets/finaltest.csv')

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

train_pts = train_data.drop('Activity', axis=1)
train_labels = train_data['Activity']

test_pts = test_data.drop('Activity', axis=1)
test_labels = test_data['Activity']


# Classifying and plotting after applying PCA
pca = PCA(n_components=10)
print("PCA start: ",time.clock())
train_pca = pca.fit_transform(train_pts, y=train_labels)
print("PCA stop: ",time.clock())
print(pca.explained_variance_ratio_.sum())
print("PCA trans start: ",time.clock())
test_pca = pca.transform(test_pts)
print("PCA trans stop: ",time.clock())
clf = classify(train_pca, train_labels, test_pca, test_labels, "PCA(RBF)")


# y_pred = clf.predict(test_pca)
# target_names = ['STANDING', 'SITTING', 'LYING', 'WALKING', 'WALKING_DOWNSTAIRS',
#                 'WALKING_UPSTAIRS']

# print(classification_report(test_labels, y_pred, target_names=target_names))
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
