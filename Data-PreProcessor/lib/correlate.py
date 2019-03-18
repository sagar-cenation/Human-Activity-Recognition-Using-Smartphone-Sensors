import pandas as pd
import numpy as np
from columns import columns

myTrain = pd.read_csv("sets2/ATrain2x.csv")
myTrain = myTrain.drop("subject",axis=1)
myTrain = myTrain.drop("Activity",axis=1)
actualTrain = pd.read_csv("sets2/train.csv")
actualTrain = actualTrain.drop("subject",axis=1)
actualTrain = actualTrain.drop("Activity",axis=1)

myCorr = pd.DataFrame(index=columns,columns=["Correlation"])

for col in myTrain.columns.values:
    print(col)
    myCorr["Correlation"][col] = np.corrcoef(myTrain[col],actualTrain[col])[0][1]

myCorr.to_csv("sets2/corrs1.csv",sep=",")
# print(myCorr)
# print(myTrain.shape)
# print(actualTrain.shape)