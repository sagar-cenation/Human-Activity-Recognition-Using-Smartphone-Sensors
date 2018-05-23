import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as cf
from sklearn.cross_validation import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('newdatasets/train.csv')
test_data = pd.read_csv('newdatasets/test.csv')
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

train_pts = train_data.drop('Activity', axis=1)
train_labels = train_data['Activity']

test_pts = test_data.drop('Activity', axis=1)
test_labels = test_data['Activity']

pca = PCA(n_components=200)
train_pca = pca.fit_transform(train_pts, y=train_labels)
# print(pca.explained_variance_ratio_)
test_pca = pca.transform(test_pts)
print(pca.explained_variance_ratio_.sum())


# plotting  confusion matrix
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
    res_nor = np.ndarray((6, 6), dtype=float)
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

rf = RandomForestClassifier(n_estimators=200, n_jobs=4, min_samples_leaf=10)  # 0.90566677977604348
rf.fit(train_pca, train_labels)
print rf.score(test_pca, test_labels)
plot_confusion(rf, test_pca, test_labels)




# precision vs recall
rf = RandomForestClassifier(n_estimators=200, n_jobs=4, min_samples_leaf=10)  # 0.90566677977604348
rf.fit(train_pca, train_labels)
y_pred = rf.predict(test_pca)
print rf.score(test_pca, test_labels)

# print y_pred
# print test_pca
target_names = ['STANDING', 'SITTING', 'LYING', 'WALKING', 'WALKING_DOWNSTAIRS',
                'WALKING_UPSTAIRS']

print(classification_report(test_labels, y_pred, target_names=target_names))
