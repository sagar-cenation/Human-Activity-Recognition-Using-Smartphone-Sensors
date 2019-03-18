import numpy as np
import scipy as sc
import scipy.stats as stat
import math
import pandas as pd

class StatFunc:

    def mad(self,X,mean):
        sum = float(0)
        for a in X:
            sum = sum + abs(a-mean)
        return (sum/len(X))

    def sma(self, X, Y, Z):
        ans = []
        for i in range(0,len(X)):
            ans.append(abs(X[i]) + abs(Y[i]) + abs(Z[i]))
        return (sum(ans)/len(X))

    def energy(self,X):
        sum = 0
        for x in X:
            sum = sum + x**2
        return sum

    # def entropy(self,X):
    #     hist = np.histogram(X,bins=len(X))
    #     hist = hist[0]
    #     return stat.entropy(hist)

    def arburg(self, X, order):
        """This version is 10 times faster than arburg, but the output rho is not correct.
        
        
        returns [1 a0,a1, an-1]
        
        """
        x = np.array(X)
        N = len(x)

        if order == 0.: 
            raise ValueError("order must be > 0")

        # Initialisation
        # ------ rho, den
        rho = sum(abs(x)**2.) / N  # Eq 8.21 [Marple]_
        den = rho * 2. * N 

        # ------ backward and forward errors
        ef = np.zeros(N)
        eb = np.zeros(N)    
        for j in range(0, N):  #eq 8.11
            ef[j] = x[j]
            eb[j] = x[j]

        # AR order to be stored
        a = np.zeros(1)
        a[0] = 1
        # ---- rflection coeff to be stored
        ref = np.zeros(order)

        # temp = 1.
        E = np.zeros(order+1)
        E[0] = rho

        for m in range(0, order):
            #print m
            # Calculate the next order reflection (parcor) coefficient
            efp = ef[1:]
            ebp = eb[0:-1]
            #print efp, ebp
            num = -2.* np.dot(ebp.conj().transpose(),  efp)
            den = np.dot(efp.conj().transpose(),  efp)
            den += np.dot(ebp,  ebp.conj().transpose())
            ref[m] = num / den

            # Update the forward and backward prediction errors
            ef = efp + ref[m] * ebp
            eb = ebp + ref[m].conj().transpose() * efp

            # Update the AR coeff.
            a.resize(len(a)+1)
            a = a + ref[m] * np.flipud(a).conjugate()

            # Update the prediction error
            E[m+1] = (1 - ref[m].conj().transpose()*ref[m]) * E[m]
            #print 'REF', ref, num, den
        return a, E[-1], ref

    def maxInds(self,X):
        max = X[0]
        inds = 0
        i = 0
        for x in X:
            if x > max:
                inds = i
                max = x
            i = i + 1
        return inds
    
    # def bandsEnergy(self,X,start,end):
    #     start = start - 1
    #     sum = 0
    #     for x in range(start, end):
    #         sum = sum + X[2*x]**2 + X[2*x+1]**2
    #     return sum

    def derivative(self,X,interval=1):
        ans = []
        for i in range(1,len(X)):
            ans.append((X[i]-X[i-1])/interval)
        return ans

    def magnitude(self,X):
        return np.sqrt(sum(a**2 for a in X))

    def magnitudeVector(self,windowX,windowY,windowZ):
        ans = list()
        for i in range(0,len(windowX)):
            ans.append(np.sqrt(windowX[i]**2 + windowY[i]**2 + windowZ[i]**2))
        return ans

    def angle(self, X, Y):
        ans = math.acos(
            np.dot(X,Y)/(self.magnitude(X) * self.magnitude(Y))
        )
        return ans

# YLabels = pd.read_csv("sets2/y_train.txt",delim_whitespace=True,names=[0])
# print(YLabels[0][0:3])
# a = list([1,2,3])
# print(a)
# bodyAccX = pd.read_csv("sets2/body_acc_x_train.txt",delim_whitespace=True,names=range(0,128))
# bodyAccY = pd.read_csv("sets2/body_acc_y_train.txt",delim_whitespace=True,names=range(0,128))
# bodyAccZ = pd.read_csv("sets2/body_acc_z_train.txt",delim_whitespace=True,names=range(0,128))

# totalAccX = pd.read_csv("sets2/total_acc_x_train.txt",delim_whitespace=True,names=range(0,128))
# totalAccY = pd.read_csv("sets2/total_acc_y_train.txt",delim_whitespace=True,names=range(0,128))
# totalAccZ = pd.read_csv("sets2/total_acc_z_train.txt",delim_whitespace=True,names=range(0,128))

# gravityX = totalAccX - bodyAccX
# gravityY = totalAccY - bodyAccY
# gravityZ = totalAccZ - bodyAccZ

# gyroX = pd.read_csv("sets2/body_gyro_x_train.txt",delim_whitespace=True,names=range(0,128))
# gyroY = pd.read_csv("sets2/body_gyro_y_train.txt",delim_whitespace=True,names=range(0,128))
# gyroZ = pd.read_csv("sets2/body_gyro_z_train.txt",delim_whitespace=True,names=range(0,128))

# subject = pd.read_csv("sets2/subject_train.txt",delim_whitespace=True,names=[0])
# YLabels = pd.read_csv("sets2/y_train.txt",delim_whitespace=True,names=[0])
# # actualTrain = pd.read_csv("sets2/train.csv")

# bodyAccX = bodyAccX.T
# bodyAccY = bodyAccY.T
# bodyAccZ = bodyAccZ.T

# gravityX = gravityX.T
# gravityY = gravityY.T
# gravityZ = gravityZ.T

# gyroX = gyroX.T
# gyroY = gyroY.T
# gyroZ = gyroZ.T

# fun = StatFunc()

# res = []

# for i in range(0,len(bodyAccX.columns.values)):
#     # a = fun.mad(gyroZ[i],np.mean(gyroZ[i]))
#     # res.append(a)
#     # break

# max = np.max(res)
# min = np.min(res)
# delta = max - min
# nor = []
# # normailze
# for i in range(0,len(res)):
#     nor.append(-1+((res[i]-min)*2/delta))

# print(nor)
# # actual = actualTrain["fBodyBodyGyroJerkMag-min()"].tolist()
# # print("Correlation: ",np.corrcoef(nor,actual)[0][1])
# # plotCorr(nor, actual)