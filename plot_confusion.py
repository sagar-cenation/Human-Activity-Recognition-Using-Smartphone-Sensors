import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import confusion_matrix as cf


def plot_confusion(classifier, test_pts, test_labels):
    classes = ['STANDING',
               'SITTING',
               'LAYING',
               'WALKING',
               'WALKING_DOWN',
               'WALKING_UP']
    
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
    cax = ax.matshow(result)
    # plt.matshow(result)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + cl)
    ax.set_yticklabels([''] + cl)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.legend(loc='best')
    plt.show()

# plot_confusion(clf, test_pts, test_labels)
