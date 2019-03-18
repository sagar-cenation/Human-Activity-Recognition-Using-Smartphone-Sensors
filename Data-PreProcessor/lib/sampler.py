import pandas as pd
import numpy as np
import math
from DataGenerator import *
import StatFunc

class WindowSampler:
    """
    This is a window sampler class which will apply fixed width window sampling over
    the given dataset as input
    """
    
    def __init__(self,frame,length,slide,startIndex,endIndex):
        self.frame = frame
        self.length = length
        self.slide = slide
        self.windowCount = int(math.floor((frame.shape[0]-length)/slide))
        self.gap = int(math.floor((frame.shape[0] - ((self.windowCount-1)*slide + length))/2))
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.tBodyAcc = Time3DStat(self.frame["AccX"].tolist(),
                                    self.frame["AccY"].tolist(),
                                    self.frame["AccZ"].tolist())
        self.tBodyAcc = Time3DStat(self.frame["AccX"].tolist(),
                                    self.frame["AccY"].tolist(),
                                    self.frame["AccZ"].tolist())
        

    def generateSamples(self):
        start = self.startIndex + self.gap
        end = start + self.length
        for i in range(0,self.windowCount):
            print(str(start) + "," + str(end))
            # TODO put the code to input the samples for generating features


            start = start + self.slide
            end = end + self.slide
        

# accData = pd.read_csv("acc_exp01_user01.txt", delim_whitespace=True, names=["AccX","AccY","AccZ"])
# gyroData = pd.read_csv("gyro_exp01_user01.txt", delim_whitespace=True, names=["GyroX","GyroY","GyroZ"])
# final_data = accData.join(gyroData)

# sample = WindowSampler(final_data.loc[250:1232],128,64,250,1232)
# print(sample.windowCount)
# print(sample.gap)
# sample.generateSamples()