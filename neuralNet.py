import numpy as np 
import pickle
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_mldata
from numpy import arange
from sklearn.utils import shuffle
import random
import scipy.misc # to visualize only
from PIL import Image
import scipy.special as ss 
import sys


#sys.stdout = open('Train.txt','w') 



class Success:


	numOfHiddenLayers = 0
	W = np.random.randn(1,1,1)
	numOfHiddenUnits = None
	numOfHiddenUnits2 = None
	W1 = None 
	W2 = None
	W3 = None
	X = None
	y = None
	y_1col = None
	numOfExamples_n = 0
	numOfFeatures_m = 0
	numOfOutputUnits = 0
	WeightsForBias1 = None
	WeightsForBias2 = None
	learningRate = 0.01
	lambdaReg = 0.03
	X_test = None
	y_test = None

	def _init_(self,numOfHiddenLayers,X,y,numOfHiddenUnits,numberOfHiddenUnits2,numOfOutputUnits,X_test,y_test):
		np.random.seed(1)
		#print('constructor')
		self.numOfHiddenUnits = numOfHiddenUnits
		self.numOfHiddenUnits2 = numberOfHiddenUnits2
		self.numOfExamples_n = len(y)
		self.numOfFeatures_m = len(X[0])
		self.numOfOutputUnits = numOfOutputUnits
		self.numOfHiddenLayers = numOfHiddenLayers
		self.X = np.insert(X, 0, 1, axis=1)
		self.X_test = np.insert(X_test, 0, 1, axis=1)
		self.y_test = y_test
		#print(self.X[0:10,0])
		self.y = y
		self.y_1col = y
		
		self.W1 = 2 * np.random.random((self.numOfHiddenUnits,self.numOfFeatures_m + 1)) - 1 # not including weight for bias
		#print('Shape of W1',np.shape(self.W1))
		self.W2 = 2 * np.random.random((self.numOfOutputUnits,self.numOfHiddenUnits2 + 1)) -1

		self.W3 = 2 * np.random.random((self.numOfHiddenUnits2,self.numOfHiddenUnits + 1)) - 1
		#print('Shape of W2',np.shape(self.W2))
		#print('Shape of W3',np.shape(self.W3))
		self.WeightsForBias1 = np.zeros((1,self.numOfHiddenUnits))
		#print('Weight for bias',self.WeightsForBias1)
		self.WeightsForBias2 = np.zeros((1,self.numOfOutputUnits))
		#print(y[0],y[1])


	def yOneHotEncoding(self,y):
		
		yOutput = np.zeros(self.numOfOutputUnits * len(y))
		yOutput = np.reshape(yOutput,(len(y),self.numOfOutputUnits))
		i = 0
		for i in range(len(y)):
			yOutput[i][y[i]] = 1

		yOutput = yOutput.astype(int)
#		print(yOutput[0]) 
#		print(yOutput[1])
		return yOutput


	def predict(self):
		self.y_test = self.yOneHotEncoding(self.y_test)
		A1 = self.X_test
		Z2 = np.dot(A1,np.transpose(self.W1))
		A2 = ss.expit(Z2)
		A2 = np.insert(A2,0,1,axis = 1)
		Z3 = np.dot(A2,np.transpose(self.W2))
		A3 = ss.expit(Z3) 
		H = A3

		#print(np.shape(y_test), np.shape(X_test))

		J = (-1 / len(self.X_test)) * (np.sum(self.y_test * np.log(H) + (1 - self.y_test) * np.log(1 - H)))
		regularizationTerm = (self.lambdaReg / (2 * len(self.X_test))) * (np.sum(self.W1[1:,:] ** 2) + np.sum(self.W2[1:,:] ** 2))	
		#print(J,regularizationTerm)
		return (J + regularizationTerm)



	def predictSum(self,Xdata):


		#self.W1 = np.loadtxt('Weights/W1.txt35000')
		#self.W2 = np.loadtxt('Weights/W2.txt35000')
		
		A1 = self.X
		Z2 = np.dot(A1,np.transpose(self.W1))
		A2 = ss.expit(Z2)
		A2 = np.insert(A2,0,1,axis = 1)
		Z3 = np.dot(A2,np.transpose(self.W3))
		A3 = ss.expit(Z3)
		A3 = np.insert(A3,0,1,axis = 1)
		Z4 = np.dot(A3,np.transpose(self.W2))
		A4 = ss.expit(Z4) 
		H = A4
		output = np.argmax(H,axis=1)
		return output

	def feedForward(self):

		self.y = self.yOneHotEncoding(self.y)
		JTrain,JTest = [],[]
		for i in range(15001):
			A1 = self.X
			Z2 = np.dot(A1,np.transpose(self.W1))
			A2 = ss.expit(Z2)
			A2 = np.insert(A2,0,1,axis = 1)
			Z3 = np.dot(A2,np.transpose(self.W3))
			A3 = ss.expit(Z3)
			A3 = np.insert(A3,0,1,axis = 1)
			Z4 = np.dot(A3,np.transpose(self.W2))
			A4 = ss.expit(Z4) 
			H = A4

			
			#----------------BACKPROPAGATION-----------
			W2Back = self.W2[:,1:]
			W3Back = self.W3[:,1:]
			del4 = H - self.y
			d3 = np.dot(del4,W2Back) * (ss.expit(Z3) * (1 - ss.expit(Z3)))
			self.W1[:,0] = 0
			self.W2[:,0] = 0
			self.W3[:,0] = 0
			del3 = np.dot(np.transpose(del4),A3)
			del2 = np.dot(np.transpose(d3),A2)
			d2 = np.dot(d3,(W3Back)) * ss.expit(Z2) * (1-ss.expit(Z2))
			del1 = np.dot(np.transpose(d2),A1)
			W1_grad = (1/self.numOfExamples_n) * del1 + (self.lambdaReg / self.numOfExamples_n) * self.W1
			W2_grad = (1/self.numOfExamples_n) * del3 + (self.lambdaReg / self.numOfExamples_n) * self.W2
			W3_grad = (1/self.numOfExamples_n) * del2 + (self.lambdaReg / self.numOfExamples_n) * self.W3

			self.W1 = self.W1 - self.learningRate * W1_grad
			self.W2 = self.W2 - self.learningRate * W2_grad
			self.W3 = self.W3 - self.learningRate * W3_grad




			J = (-1 / self.numOfExamples_n) * (np.sum(self.y * np.log(H) + (1 - self.y) * np.log(1 - H)))
			regularizationTerm = (self.lambdaReg / (2 * self.numOfExamples_n)) * (np.sum(self.W1[1:,:] ** 2) + np.sum(self.W2[1:,:] ** 2))	
			#print(J,regularizationTerm)
			JTrain.append(J + regularizationTerm)
			#JTest.append( self.predict())
			if i % 500 == 0:
				print('----------------------------',i,'--------------------')
				#print('\nJTrain\n',JTrain)
				#print('\nJTest\n',JTest)
				fname1 = 'W1.txt' + str(i)
				fname2 = 'W2.txt' + str(i)
				np.savetxt(fname1,self.W1)
				np.savetxt(fname2,self.W2)



	def feedForwardTrainMore(self):

		self.y = self.yOneHotEncoding(self.y)
		JTrain,JTest = [],[]
		for i in range(15001):
			A1 = self.X
			Z2 = np.dot(A1,np.transpose(self.W1))
			A2 = ss.expit(Z2)
			A2 = np.insert(A2,0,1,axis = 1)
			Z3 = np.dot(A2,np.transpose(self.W3))
			A3 = ss.expit(Z3)
			A3 = np.insert(A3,0,1,axis = 1)
			Z4 = np.dot(A3,np.transpose(self.W2))
			A4 = ss.expit(Z4) 
			H = A4

			print('shape',np.shape(H))
			#pseudoTheta2 = Theta2(:,2:end);
			W2Back = self.W2[:,1:]
			W3Back = self.W3[:,1:]
			del4 = H - self.y
			d3 = np.dot(del4,W2Back) * (ss.expit(Z3) * (1 - ss.expit(Z3)))
			self.W1[:,0] = 0
			self.W2[:,0] = 0
			self.W3[:,0] = 0
			del3 = np.dot(np.transpose(del4),A3)
			del2 = np.dot(np.transpose(d3),A2)
			d2 = np.dot(d3,(W3Back)) * ss.expit(Z2) * (1-ss.expit(Z2))
			del1 = np.dot(np.transpose(d2),A1)
			W1_grad = (1/self.numOfExamples_n) * del1 + (self.lambdaReg / self.numOfExamples_n) * self.W1
			W2_grad = (1/self.numOfExamples_n) * del3 + (self.lambdaReg / self.numOfExamples_n) * self.W2
			W3_grad = (1/self.numOfExamples_n) * del2 + (self.lambdaReg / self.numOfExamples_n) * self.W3

			self.W1 = self.W1 - self.learningRate * W1_grad
			self.W2 = self.W2 - self.learningRate * W2_grad
			self.W3 = self.W3 - self.learningRate * W3_grad




			#J = (-1 / self.numOfExamples_n) * (np.sum(self.y * np.log(H) + (1 - self.y) * np.log(1 - H)))
			#regularizationTerm = (self.lambdaReg / (2 * self.numOfExamples_n)) * (np.sum(self.W1[1:,:] ** 2) + np.sum(self.W2[1:,:] ** 2))	
			#print(J,regularizationTerm)
			#JTrain.append(J + regularizationTerm)
			#JTest.append( self.predict())
			if i % 500 == 0:
				print('----------------------------',i,'--------------------')
				#print('\nJTrain\n',JTrain)
				#print('\nJTest\n',JTest)
				fname1 = 'W1.txt' + str(i)
				fname2 = 'W2.txt' + str(i)
				np.savetxt(fname1,self.W1)
				np.savetxt(fname2,self.W2)


#end of class------------


mnist = fetch_mldata('MNIST original')

n_train = 60000
n_test = 10000

# Define training and testing sets
indices = arange(len(mnist.data))
random.seed(0)
#train_idx = random.sample(indices, n_train)
#test_idx = random.sample(indices, n_test)
train_idx = arange(0,n_train)
test_idx = arange(n_train+1,n_train+n_test)

X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

#print(np.shape(X_train),"	",type(X_train))
#print(np.shape(y_train),"	",type(y_train))
np.reshape(y_train,(60000,1))

X_train = X_train.astype(int)
y_train = y_train.astype(int)
X_test  = X_test.astype(int)
y_test  = y_test.astype(int)

X_train = X_train // 210
X_test = X_test // 210

X_train, y_train = shuffle(X_train, y_train, random_state=0)
#plt.imshow(np.reshape(X_train[1],(28,28)))
plt.show()
obj = Success()
obj._init_(100,X_train,y_train,100,100,10,X_test,y_test)
#obj.yOneHotEncoding(y_train)
obj.feedForward()
a = np.array([[1,2],[5,6]])
b = np.array([[10,30],[3,4]])

#load project data


op = obj.predictSum(X_test)
loss = np.subtract(op,y_test) 
missed = np.count_nonzero(loss)
print('The num of missed ',missed)


Xdata = pickle.load(open('X.p','rb'))
Ydata = pickle.load(open('Ys.p', 'rb'))


Xdata = np.divide(Xdata, 255)
#print('Shape of Xdata ',np.shape(Xdata))
#print('Shape of Ydata ',np.shape(Ydata))
Xdata0 = Xdata[0] 
Xdata0 = np.reshape(Xdata0,(10000,-1))
Xdata1 = Xdata[1] 
Xdata1 = np.reshape(Xdata1,(10000,-1))

#print('Shape of Xdata ',np.shape(Xdata0))
#print('Shape of Xdata ',np.shape(Xdata1))

first = obj.predictSum(Xdata0)
second = obj.predictSum(Xdata1)

predSum = first + second

loss = np.subtract(predSum,Ydata) 
missed = np.count_nonzero(loss)
print('The num of missed ',missed)


#print('XTrain \n',X_train[15])
#print('Xdata0\n',Xdata0[15])

#obj.feedForwardTrainMore()

#print(a * b)
#print(np.multiply(a,b))