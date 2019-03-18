from StatFunc import StatFunc
import pandas as pd
import numpy as np
import math

# arr = [0.02**3,0.04**3,0.06**3,0.08**3,0.1**3,0.12**3,0.14**3,0.16**3,0.18**3,0.2**3]
# fun = StatFunc().doubleDerivative(arr,0.02)
# print(fun)

# fun = StatFunc()
# arr = [1,4,8,16,32,64,128,256,512,1024,2048]
# ans,x,y = fun.arburg(arr,4)
# print(ans)

train_data = pd.read_csv("lib/train.csv")
# madSum = train_data['tBodyAcc-std()-X'] + train_data['tBodyAcc-std()-Y'] + train_data['tBodyAcc-std()-Z']
# print(np.corrcoef(train_data['tBodyAcc-sma()'],madSum))

# X = [1,3,5,-2,6,-9,11,-16]
# Y = []
# for x in X:
#     Y.append(abs(x))
# print(Y)
# print(sum(Y))
# bodyAccMeanX = train_data['tBodyAcc-mean()-X']
# bodyAccMeanY = train_data['tBodyAcc-mean()-Y']
# bodyAccMeanZ = train_data['tBodyAcc-mean()-Z']
gravAccMeanX = train_data['tGravityAcc-mean()-X']
gravAccMeanY = train_data['tGravityAcc-mean()-Y']
gravAccMeanZ = train_data['tGravityAcc-mean()-Z']
angle = []
# for i in range(0,len(bodyAccMeanX)):
#     arr1 = [bodyAccMeanX[i], bodyAccMeanY[i], bodyAccMeanZ[i]]
#     arr2 = [gravAccMeanX[i], gravAccMeanY[i], gravAccMeanZ[i]]
#     angle.append(math.acos(np.dot(arr1,arr2)/(
#         np.sqrt(sum(a**2 for a in arr1))*np.sqrt(sum(a**2 for a in arr2)))))

# # print(angle)
# print(np.corrcoef(angle,train_data["angle(tBodyAccMean,gravity)"]))

for i in range(0,len(gravAccMeanX)):
    arr2 = [gravAccMeanX[i], gravAccMeanY[i], gravAccMeanZ[i]]
    angle.append(
        math.acos(gravAccMeanZ[i]/np.sqrt(sum(a**2 for a in arr2)))
    )

# print(angle)
print("For Z")
print(np.corrcoef(angle,train_data["angle(Z,gravityMean)"]))