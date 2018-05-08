import numpy as np
import pandas as pd

train = pd.read_csv("datasets/finaltrain.csv")
test = pd.read_csv("datasets/finaltest.csv")

nortrain = pd.DataFrame()
nortest = pd.DataFrame()

for i in train.columns.values:
    max = np.max(train[i].tolist()+test[i].tolist())
    min = np.min(train[i].tolist()+test[i].tolist())