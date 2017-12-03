import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import confusion_matrix as cf
from sklearn import svm

def classify(train, train_labels, test, test_labels, method, SVMtype):
    clf = svm.SVC(decision_function_shape=SVMtype)
    clf.fit(train,train_labels)
    print("Accuracy in ", method, " = ", clf.score(test,test_labels))
    return clf

def plot_confusion(classifier, test_pts, test_labels):
    classes = ['STANDING',
    'SITTING',
    'LAYING',
    'WALKING',
    'WALKING_DOWNSTAIRS',
    'WALKING_UPSTAIRS']
    pred_label = classifier.predict(test_pts)
    #print(true_label)
    result = cf(test_labels, pred_label, labels=classes)
    print(result)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(result)
    #plt.matshow(result)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.legend(loc='best')
    plt.show()

train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

train_pts = train_data.drop('Activity', axis=1)
train_labels = train_data['Activity']

test_pts = test_data.drop('Activity', axis=1)
test_labels = test_data['Activity']


# Classifying and plotting after applying PCA
#pca = PCA(n_components=200)
#train_pca = pca.fit_transform(train_pts)
#print(pca.explained_variance_ratio_)
#test_pca = pca.fit_transform(test_pts)
#print(pca.explained_variance_ratio_)
#ovoclf = classify(train_pca, train_labels, test_pca, test_labels, "PCA(ovr)", 'ovo')
#plot_confusion(ovoclf, test_pca, test_labels)

# Classifying and plotting Actual dataset
ovoclf = classify(train_pts, train_labels, test_pts, test_labels, "Normal", 'ovr')
plot_confusion(ovoclf, test_pts, test_labels)