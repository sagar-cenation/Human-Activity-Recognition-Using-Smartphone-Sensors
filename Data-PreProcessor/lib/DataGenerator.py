# time3Dstat
# time3dmag
# fft3dstat
# fft3dmag
import numpy as np
import scipy.stats as stats
from lib.StatFunc import StatFunc

class Time3DStat:
    
    def __init__(self,windowX,windowY,windowZ):
        self.windowX = windowX
        self.windowY = windowY
        self.windowZ = windowZ

    def generate(self):
        fun = StatFunc()
        ans = [np.mean(self.windowX),
            np.mean(self.windowY),
            np.mean(self.windowZ),
            np.std(self.windowX),
            np.std(self.windowY),
            np.std(self.windowZ),
        ]
        # print(ans[0])
        ans2 = [fun.mad(self.windowX,ans[0]),
            fun.mad(self.windowY,ans[1]),
            fun.mad(self.windowZ,ans[2]),
            np.max(self.windowX),
            np.max(self.windowY),
            np.max(self.windowZ),
            np.min(self.windowX),
            np.min(self.windowY),
            np.min(self.windowZ),
            fun.sma(self.windowX,self.windowY,self.windowZ),
            fun.energy(self.windowX),
            fun.energy(self.windowY),
            fun.energy(self.windowZ),
            stats.iqr(self.windowX),
            stats.iqr(self.windowY),
            stats.iqr(self.windowZ),
            # fun.entropy(self.windowX),
            # fun.entropy(self.windowY),
            # fun.entropy(self.windowZ),
        ]
        ans = ans + ans2
        arc = fun.arburg(self.windowX,4)
        ans = ans + [arc[0][1], arc[0][2], arc[0][3], arc[0][4]]
        arc = fun.arburg(self.windowY,4)
        ans = ans + [arc[0][1], arc[0][2], arc[0][3], arc[0][4]]
        arc = fun.arburg(self.windowZ,4)
        ans = ans + [arc[0][1], arc[0][2], arc[0][3], arc[0][4]]
        corr = np.corrcoef(self.windowX,self.windowY)
        ans.append(corr[0][1])
        corr = np.corrcoef(self.windowX,self.windowZ)
        ans.append(corr[0][1])
        corr = np.corrcoef(self.windowY,self.windowZ)
        ans.append(corr[0][1])

        return ans

class Time3DMag:
    def __init__(self,window):
        self.window = window

    def generate(self):
        fun = StatFunc()
        ans = [
            np.mean(self.window),
            np.std(self.window),
        ]
        ans2 = [
            fun.mad(self.window,ans[0]),
            np.max(self.window),
            np.min(self.window),
            ans[0],
            fun.energy(self.window),
            stats.iqr(self.window),
            # fun.entropy(self.window)
        ]
        ans = ans + ans2
        arc = fun.arburg(self.window,4)
        ans = ans + [arc[0][1], arc[0][2], arc[0][3], arc[0][4]]
        return ans

class FFT3DStat:

    def __init__(self,windowX,windowY,windowZ):
        self.windowX = windowX
        self.windowY = windowY
        self.windowZ = windowZ

    def generate(self):
        fun = StatFunc()
        ans = [
            np.mean(self.windowX),
            np.mean(self.windowY),
            np.mean(self.windowZ),
            np.std(self.windowX),
            np.std(self.windowY),
            np.std(self.windowZ),
        ]
        ans2 = [
            fun.mad(self.windowX,ans[0]),
            fun.mad(self.windowY,ans[1]),
            fun.mad(self.windowZ,ans[2]),
            np.max(self.windowX),
            np.max(self.windowY),
            np.max(self.windowZ),
            np.min(self.windowX),
            np.min(self.windowY),
            np.min(self.windowZ),
            fun.sma(self.windowX,self.windowY,self.windowZ),
            fun.energy(self.windowX),
            fun.energy(self.windowY),
            fun.energy(self.windowZ),
            stats.iqr(self.windowX),
            stats.iqr(self.windowY),
            stats.iqr(self.windowZ),
            # fun.entropy(self.windowX),
            # fun.entropy(self.windowY),
            # fun.entropy(self.windowZ),
            fun.maxInds(self.windowX),
            fun.maxInds(self.windowY),
            fun.maxInds(self.windowZ),
            np.average(range(1,len(self.windowX)+1),weights=self.windowX),
            np.average(range(1,len(self.windowY)+1),weights=self.windowY),
            np.average(range(1,len(self.windowZ)+1),weights=self.windowZ),
            stats.skew(self.windowX),
            stats.kurtosis(self.windowX),
            stats.skew(self.windowY),
            stats.kurtosis(self.windowY),
            stats.skew(self.windowZ),
            stats.kurtosis(self.windowZ)
            # fun.bandsEnergy(self.windowX,1,8),
            # fun.bandsEnergy(self.windowX,9,16),
            # fun.bandsEnergy(self.windowX,17,24),
            # fun.bandsEnergy(self.windowX,25,32),
            # fun.bandsEnergy(self.windowX,33,40),
            # fun.bandsEnergy(self.windowX,41,48),
            # fun.bandsEnergy(self.windowX,49,56),
            # fun.bandsEnergy(self.windowX,57,64),
            # fun.bandsEnergy(self.windowX,1,16),
            # fun.bandsEnergy(self.windowX,17,32),
            # fun.bandsEnergy(self.windowX,33,48),
            # fun.bandsEnergy(self.windowX,49,64),
            # fun.bandsEnergy(self.windowX,1,24),
            # fun.bandsEnergy(self.windowX,25,48),
            # fun.bandsEnergy(self.windowY,1,8),
            # fun.bandsEnergy(self.windowY,9,16),
            # fun.bandsEnergy(self.windowY,17,24),
            # fun.bandsEnergy(self.windowY,25,32),
            # fun.bandsEnergy(self.windowY,33,40),
            # fun.bandsEnergy(self.windowY,41,48),
            # fun.bandsEnergy(self.windowY,49,56),
            # fun.bandsEnergy(self.windowY,57,64),
            # fun.bandsEnergy(self.windowY,1,16),
            # fun.bandsEnergy(self.windowY,17,32),
            # fun.bandsEnergy(self.windowY,33,48),
            # fun.bandsEnergy(self.windowY,49,64),
            # fun.bandsEnergy(self.windowY,1,24),
            # fun.bandsEnergy(self.windowY,25,48),
            # fun.bandsEnergy(self.windowZ,1,8),
            # fun.bandsEnergy(self.windowZ,9,16),
            # fun.bandsEnergy(self.windowZ,17,24),
            # fun.bandsEnergy(self.windowZ,25,32),
            # fun.bandsEnergy(self.windowZ,33,40),
            # fun.bandsEnergy(self.windowZ,41,48),
            # fun.bandsEnergy(self.windowZ,49,56),
            # fun.bandsEnergy(self.windowZ,57,64),
            # fun.bandsEnergy(self.windowZ,1,16),
            # fun.bandsEnergy(self.windowZ,17,32),
            # fun.bandsEnergy(self.windowZ,33,48),
            # fun.bandsEnergy(self.windowZ,49,64),
            # fun.bandsEnergy(self.windowZ,1,24),
            # fun.bandsEnergy(self.windowZ,25,48)
        ]
        ans = ans + ans2

        return ans

class FFT3DMag:
    def __init__(self,window):
        self.window = window

    def generate(self):
        fun = StatFunc()
        ans = [
            np.mean(self.window),
            np.std(self.window),
        ]
        ans2 = [
            fun.mad(self.window,ans[0]),
            np.max(self.window),
            np.min(self.window),
            ans[0],
            fun.energy(self.window),
            stats.iqr(self.window),
            # fun.entropy(self.window),
            fun.maxInds(self.window),
            np.average(range(1,len(self.window)+1),weights=self.window),
            stats.skew(self.window),
            stats.kurtosis(self.window)
        ]
        ans = ans + ans2

        return ans