import pandas as pd
import matplotlib.pyplot as plt

def plotGraph(train,fx,fy):
	div = train.groupby('Activity')
	for name,group in div:
 		plt.plot(group[fx],group[fy],'.',label=name)

	plt.ylabel(fy)
	plt.xlabel(fx)
	plt.title(fy + " V/S " + fx)
	plt.legend(loc='best')
	plt.show()

train = pd.read_csv('train.csv')

plotGraph(train,'tGravityAcc-mean()-X','tGravityAcc-max()-X')