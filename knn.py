import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix as cf
from sklearn.cross_validation import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def classify(train, train_labels, test, test_labels, method):
    prop = {
        'n_neighbors': None,
        'weights': None,
        'algorithm': None,
        'p': None
    }

    k_range = list(range(5, 12))
    k_scores = []
    for n_neighbors in k_range:
        for weights in ('uniform','distance',)
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train, train_labels)
        # scores = clf.score(train, train_labels)
        # k_scores.append(scores)
        scores = cross_val_score(clf, test, test_labels, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    return k_scores


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


# Partitioning data to input and target variable
train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

train_pts = train_data.drop('Activity', axis=1)
train_labels = train_data['Activity']

test_pts = test_data.drop('Activity', axis=1)
test_labels = test_data['Activity']


# Classifying and plotting after applying PCA
pca = PCA(n_components=200)
train_pca = pca.fit_transform(train_pts, y=train_labels)
# print(pca.explained_variance_ratio_)
test_pca = pca.transform(test_pts)
# print(pca.explained_variance_ratio_)

k_range = list(range(1, 31))
k_scores = classify(train_pca, train_labels, test_pca, test_labels, "PCA")
print(k_scores)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('KNN Accuracy')
plt.show()


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
