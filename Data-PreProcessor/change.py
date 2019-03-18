import pandas as pd
import numpy as np
# from columns import columns
from lib.DataGenerator import *
from lib.StatFunc import StatFunc
import scipy.stats as stat

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

myDataset = pd.read_csv("sets2/ATrain2.csv")

changeSet = pd.DataFrame()

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

fun = StatFunc()
print(len(changeSet.columns.values))
# Change doublederivtive into single derivative in gyroscope

for i in range(0,len(bodyAccX.columns.values)):

    gyroVecX = gyroX[i].tolist()
    gyroVecY = gyroY[i].tolist()
    gyroVecZ = gyroZ[i].tolist()
    
    gyroJerkX = fun.derivative([2*gyroVecX[0]-gyroVecX[1]]+gyroVecX)
    gyroJerkY = fun.derivative([2*gyroVecY[0]-gyroVecY[1]]+gyroVecY)
    gyroJerkZ = fun.derivative([2*gyroVecZ[0]-gyroVecZ[1]]+gyroVecZ)

    # Finding magnitude
    gyroJMag = fun.magnitudeVector(gyroJerkX, gyroJerkY, gyroJerkZ)

    # Data
    tBodyGyroJerk = Time3DStat(gyroJerkX,gyroJerkY,gyroJerkZ).generate()
    tBodyGyroJerkMag = Time3DMag(gyroJMag).generate()

    # Applying FFT
    FFTgyroJerkMag = np.absolute(np.fft.fft(gyroJMag))

    # Data
    fBodyBodyGyroJerkMag = FFT3DMag(FFTgyroJerkMag).generate()

    # Setting changes into the train data
    datapoint = tBodyGyroJerk + tBodyGyroJerkMag + fBodyBodyGyroJerkMag

    # Adding into my dataset
    changeSet.loc[i] = datapoint

# Making all the changes into myTrain
for col in changeSet.columns.values:
    myDataset[col] = changeSet[col]

myDataset.to_csv("sets2/myTrain.csv",sep=",")