import pandas as pd
import numpy as np
from columns import columns
from lib.DataGenerator import *
from lib.StatFunc import StatFunc
import scipy.stats as stat
# train = pd.read_csv("lib/train.csv")

# print(len(train.columns.values))
# print(columns)
# print(len(columns))
# mylist = train["Activity"].tolist()
# print(len(mylist))
# print(mylist)
# myTrain = pd.read_csv("X_train1.txt", delim_whitespace=True)
# newTrain = pd.read_csv("X_train2.txt", delim_whitespace=True)
# print(myTrain.shape)
# print(newTrain.shape)

# f1 = pd.DataFrame(data=[[1, 2], [3, 4]])
# f2 = pd.DataFrame(data=[[10, 11], [12, 13]])
# print(f1*f2)
bodyAccX = pd.read_csv("sets2/body_acc_x_train.txt",delim_whitespace=True,names=range(0,128))
bodyAccY = pd.read_csv("sets2/body_acc_y_train.txt",delim_whitespace=True,names=range(0,128))
bodyAccZ = pd.read_csv("sets2/body_acc_z_train.txt",delim_whitespace=True,names=range(0,128))

totalAccX = pd.read_csv("sets2/total_acc_x_train.txt",delim_whitespace=True,names=range(0,128))
totalAccY = pd.read_csv("sets2/total_acc_y_train.txt",delim_whitespace=True,names=range(0,128))
totalAccZ = pd.read_csv("sets2/total_acc_z_train.txt",delim_whitespace=True,names=range(0,128))

gravityX = totalAccX - bodyAccX
gravityY = totalAccY - bodyAccY
gravityZ = totalAccZ - bodyAccZ

gyroX = pd.read_csv("sets2/body_gyro_x_train.txt",delim_whitespace=True,names=range(0,128))
gyroY = pd.read_csv("sets2/body_gyro_y_train.txt",delim_whitespace=True,names=range(0,128))
gyroZ = pd.read_csv("sets2/body_gyro_z_train.txt",delim_whitespace=True,names=range(0,128))

subject = pd.read_csv("sets2/subject_train.txt",delim_whitespace=True,names=[0])
YLabels = pd.read_csv("sets2/y_train.txt",delim_whitespace=True,names=[0])

myDataset = pd.DataFrame(columns=columns)

Activity = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Transpose all the frames
bodyAccX = bodyAccX.T
bodyAccY = bodyAccY.T
bodyAccZ = bodyAccZ.T

gravityX = gravityX.T
gravityY = gravityY.T
gravityZ = gravityZ.T

gyroX = gyroX.T
gyroY = gyroY.T
gyroZ = gyroZ.T

# TODO funcs left
# arcoeff
# entropy
# jerk(derivative)
# gyro-mean-XYZ
# gyro-jerk(doubleDerivative)
# gravityMag
# gravity-mean
# skewness
# kurtosis
####################
## For fft, f = absolute(fft(X))
## min function over fourier gives perfect results
## in fourier, corr(mean,sma) = 1
## Slight distortion in angles
## Awful readings of angle(Y,gravMean)
## ArCoeff - (tbodyAcc-X)1,2 - Distorted, 3 - perfect, 4 - Awful
## Arcoeff - (tbodyAcc-Y)1,2,3,4 - perfect
## Arcoeff - (tbodyAcc-Z)1,2 - Nasty, 3 - distorted, 4 - perfect

fun = StatFunc()
# res = []
# f = np.absolute(np.fft.fft(bodyAccX[1]))
# print(f)
# print(np.max(f))
# fun.entropy(bodyAccX[1])
# f = np.fft.fft(bodyAccX[0])
# print(fun.bandsEnergy(f,1,8))
# for i in range(0,len(bodyAccX.columns.values)):
    
#     res.append(fun.arburg(bodyAccZ[i],4)[0][1:])
# print(res)
# res = pd.DataFrame(res)
# res = res[3]
# print(res)
# max = np.max(res)
# min = np.min(res)
# delta = max - min
# nor = []
# # normailze
# for i in range(0,len(res)):
#     nor.append(-1+((res[i]-min)*2/delta))

# print(nor)
# print(type([2*bodyAccX[0]+bodyAccX[1]] + bodyAccX.tolist()))
for i in range(0,len(bodyAccX.columns.values)):
    # Getting elemental rows
    bodyAX = bodyAccX[i].tolist()
    bodyAY = bodyAccY[i].tolist()
    bodyAZ = bodyAccZ[i].tolist()

    gravX = gravityX[i].tolist()
    gravY = gravityY[i].tolist()
    gravZ = gravityZ[i].tolist()

    gyroVecX = gyroX[i].tolist()
    gyroVecY = gyroY[i].tolist()
    gyroVecZ = gyroZ[i].tolist()

    # Finding jerk signals
    # For body acceleration
    bodyAccJerkX = fun.derivative([2*bodyAX[0]-bodyAX[1]]+bodyAX,interval=0.02)
    bodyAccJerkY = fun.derivative([2*bodyAY[0]-bodyAY[1]]+bodyAY,interval=0.02)
    bodyAccJerkZ = fun.derivative([2*bodyAZ[0]-bodyAZ[1]]+bodyAZ,interval=0.02)

    # For Gyroscope
    gyroJerkX = fun.derivative([2*gyroVecX[0]-gyroVecX[1]]+gyroVecX,interval=0.02)
    gyroJerkY = fun.derivative([2*gyroVecY[0]-gyroVecY[1]]+gyroVecY,interval=0.02)
    gyroJerkZ = fun.derivative([2*gyroVecZ[0]-gyroVecZ[1]]+gyroVecZ,interval=0.02)

    # Finding magnitudes
    bodyAccMag = fun.magnitudeVector(bodyAX, bodyAY, bodyAZ)
    gravMag = fun.magnitudeVector(gravX, gravY, gravZ)
    bodyAJMag = fun.magnitudeVector(bodyAccJerkX, bodyAccJerkY, bodyAccJerkZ)
    gyroMag = fun.magnitudeVector(gyroVecX, gyroVecY, gyroVecZ)
    gyroJMag = fun.magnitudeVector(gyroJerkX, gyroJerkY, gyroJerkZ)

    # Generating data
    tBodyAcc = Time3DStat(bodyAX, bodyAY, bodyAZ).generate()
    tGravityAcc = Time3DStat(gravX, gravY, gravZ).generate()
    tBodyAccJerk = Time3DStat(bodyAccJerkX, bodyAccJerkY, bodyAccJerkZ).generate()
    tBodyGyro = Time3DStat(gyroVecX, gyroVecY, gyroVecZ).generate()
    tBodyGyroJerk = Time3DStat(gyroJerkX,gyroJerkY,gyroJerkZ).generate()

    tBodyAccMag = Time3DMag(bodyAccMag).generate()
    tGravityAccMag = Time3DMag(gravMag).generate()
    tBodyAccJerkMag = Time3DMag(bodyAJMag).generate()
    tBodyGyroMag = Time3DMag(gyroMag).generate()
    tBodyGyroJerkMag = Time3DMag(gyroJMag).generate()

    #Applying FFT
    # For bodyAcc and bodyAccJerk
    FFTBodyAccX = np.absolute(np.fft.fft(bodyAX))
    FFTBodyAccY = np.absolute(np.fft.fft(bodyAY))
    FFTBodyAccZ = np.absolute(np.fft.fft(bodyAZ))

    FFTBodyAccJerkX = np.absolute(np.fft.fft(bodyAccJerkX))
    FFTBodyAccJerkY = np.absolute(np.fft.fft(bodyAccJerkY))
    FFTBodyAccJerkZ = np.absolute(np.fft.fft(bodyAccJerkZ))

    # For bodyGyro
    FFTBodyGyroX = np.absolute(np.fft.fft(gyroVecX))
    FFTBodyGyroY = np.absolute(np.fft.fft(gyroVecY))
    FFTBodyGyroZ = np.absolute(np.fft.fft(gyroVecZ))

    # For bodyAccMag, bodyAccJerkMag, bodyGyroMag, bodyGyroMag
    FFTbodyAccMag = np.absolute(np.fft.fft(bodyAccMag))
    FFTbodyAccJerkMag = np.absolute(np.fft.fft(bodyAJMag))
    FFTgyroMag = np.absolute(np.fft.fft(gyroMag))
    FFTgyroJerkMag = np.absolute(np.fft.fft(gyroJMag))

    # Generating data
    fBodyAcc = FFT3DStat(FFTBodyAccX, FFTBodyAccY, FFTBodyAccZ).generate()
    fBodyAccJerk = FFT3DStat(FFTBodyAccJerkX, FFTBodyAccJerkY, FFTBodyAccJerkZ).generate()
    fBodyGyro = FFT3DStat(FFTBodyGyroX, FFTBodyGyroY, FFTBodyGyroZ).generate()

    fBodyAccMag = FFT3DMag(FFTbodyAccMag).generate()
    fBodyBodyAccJerkMag = FFT3DMag(FFTbodyAccJerkMag).generate()
    fBodyBodyGyroMag = FFT3DMag(FFTgyroMag).generate()
    fBodyBodyGyroJerkMag = FFT3DMag(FFTgyroJerkMag).generate()

    # Generating angles
    AngleBodyAccGrav = fun.angle(tBodyAcc[0:3],tGravityAcc[0:3])
    AngleBodyAccJerkGrav = fun.angle(tBodyAccJerk[0:3],tGravityAcc[0:3])
    AngleBodyGyroGrav = fun.angle(tBodyGyro[0:3],tGravityAcc[0:3])
    AngleBodyGyroJerkGrav = fun.angle(tBodyGyroJerk[0:3],tGravityAcc[0:3])
    AngleGravX = fun.angle([1,0,0],tGravityAcc[0:3])
    AngleGravY = fun.angle([0,1,0],tGravityAcc[0:3])
    AngleGravZ = fun.angle([0,0,1],tGravityAcc[0:3])

    # Concatenating all the data segments
    dataPoint = tBodyAcc + tGravityAcc + tBodyAccJerk + tBodyGyro + tBodyGyroJerk
    # dataPoint.append(tBodyAccJerk).append(tBodyGyro).append(tBodyGyroJerk)
    dataPoint = dataPoint + tBodyAccMag + tGravityAccMag + tBodyAccJerkMag + tBodyGyroMag
    # dataPoint.append(tBodyAccMag).append(tGravityAccMag).append(tBodyAccJerkMag)
    # dataPoint.append(tBodyGyroMag).append(tBodyGyroJerkMag)
    dataPoint = dataPoint + tBodyGyroJerkMag + fBodyAcc + fBodyAccJerk + fBodyGyro
    # dataPoint.append(fBodyAcc).append(fBodyAccJerk).append(fBodyGyro)
    dataPoint = dataPoint + fBodyAccMag + fBodyBodyAccJerkMag + fBodyBodyGyroMag
    # dataPoint.append(fBodyAccMag).append(fBodyBodyAccJerkMag).append(fBodyBodyGyroMag)
    dataPoint = dataPoint + fBodyBodyGyroJerkMag + [AngleBodyAccGrav]
    # dataPoint.append(fBodyBodyGyroJerkMag).append(AngleBodyAccGrav)
    dataPoint = dataPoint + [AngleBodyAccJerkGrav, AngleBodyGyroGrav, AngleBodyGyroJerkGrav]
    # dataPoint.append(AngleBodyAccJerkGrav).append(AngleBodyGyroGrav).append(AngleBodyGyroJerkGrav)
    dataPoint = dataPoint + [AngleGravX, AngleGravY, AngleGravZ, subject[0][i], Activity[YLabels[0][i]]]
    # dataPoint.append(AngleGravX).append(AngleGravY).append(AngleGravZ)
    # dataPoint.append(subject[0][i]).append(Activity[YLabels[0][i]])
    # print(type(dataPoint))

    # Inserting into the dataframe
    # print(dataPoint)
    # myDataset = myDataset.append(dataPoint)
    myDataset.loc[i] = dataPoint

# print(myDataset)
myDataset.to_csv("sets2/myTrain2.csv",sep=",")
# correlation.to_csv("datasets/correlation.csv",sep=",")