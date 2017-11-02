import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix as cf

# Loading the training and testing data sets
train_data = pd.read_csv('human-activity-recognition-with-smartphones/train.csv')
test_data = pd.read_csv('human-activity-recognition-with-smartphones/test.csv')

# print test_data.head(2)

# Partitioning data to input and target variable

x_train = train_data.drop('Activity', axis=1)

y_train = pd.get_dummies(train_data.Activity)  # create dummies from a categorical column of a dataframe

x_test = test_data.drop('Activity', axis=1)
y_test = pd.get_dummies(test_data.Activity)

# finding if any missing values
missing = x_train.isnull().sum()
missing = missing[missing > 0]
# print missing

# Train Using Random Forest with 20 Trees

rf = RandomForestClassifier(20)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

cf(np.argmax(y_test.as_matrix(), axis=1), np.argmax(y_pred, axis=1))

print rf.score(x_test, y_test)

plt.matshow(cf(np.argmax(y_test.as_matrix(), axis=1), np.argmax(y_pred, axis=1)), cmap='Reds')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
