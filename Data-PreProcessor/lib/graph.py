import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotGraph(train,fx,fy):
	div = train.groupby('Activity')
	for name,group in div:
 		plt.plot(group[fx],group[fy],'.',label=name)

	plt.ylabel(fy)
	plt.xlabel(fx)
	plt.title(fy + " V/S " + fx)
	plt.legend(loc='best')
	plt.show()

# def plotGraph3D(train,fx,fy,fz):
# 	div = train.groupby('Activity')
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection="3d")
# 	for name,group in div:
#  		ax.scatter(group[fx],group[fy],group[fz],'.',label=name)
# 	ax.set_xlabel(fx)
# 	ax.set_ylabel(fy)
# 	ax.set_zlabel(fz)
# 	ax.legend()
# 	plt.show()

def plotCorr(calculated, actual):
    plt.plot(calculated,label="Calcualted")
    plt.plot(actual,label="Generated")
    plt.legend(loc="best")
    plt.show()

# train = pd.read_csv('datasets/train.csv')

# plotGraph3D(train,'tBodyAcc-mean()-X','tBodyAcc-mad()-X','tBodyAcc-arCoeff()-X,1')