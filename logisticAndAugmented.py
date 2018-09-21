import numpy
from numpy import dot
from numpy import linalg
from numpy import sqrt
import pylab
import csv
import scipy.misc # to visualize only
from matplotlib import pyplot as plt
from Queue import Queue
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, decomposition, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from PIL import Image
import random
import pickle
if True: 
		import keras
		from keras.datasets import mnist
		from keras.models import Sequential
		from keras.models import load_model
		from keras.layers import Dense
		from keras.layers import Dropout
		from keras.layers import Flatten
		from keras.layers.core import Flatten, Dense, Dropout
		from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
		from keras.optimizers import SGD
		from keras.utils import np_utils
		from keras import backend as K
model =keras.models.load_model('4layers.h5')
plan= []
def resize (params):
	jmin,jmax,kmin,kmax = params
	jdiff = (28-(jmax - jmin))
	while jdiff>=1:
		if jmin>0:
		 	jmin-=1 
			jdiff -=1
		if jdiff== 0: break 
		if jmax<59:
		 	jmax+=1 
			jdiff-=1
	kdiff = (28-(kmax - kmin))
	while kdiff>=1:
		if kmin>0:
		 	kmin-=1 
			kdiff -=1
		if kdiff== 0: break 
		if kmax<59:
		 	kmax+=1 
			kdiff-=1
	return jmin,jmax,kmin,kmax 
def top2(hull, y):
	max1 , max2 = [0,0, 0], [0,0, 0]
	c = []
	maxes = [60,0,60,0]
	q = Queue()
	count = 0 
	for p in hull: 
		m = 0
		temp = []
		if y[p[0]][p[1]]: 
			y[p[0]][p[1]] = False
			temp.append(p)
			q.put(p) 
			jmin,jmax,kmin,kmax = 60,0,60,0
			ctemp = []
			while not q.empty():
				j,k = q.get()
				ctemp.append([j,k])
				jmin = min(jmin,j)
				jmax = max(jmax,j)
				kmin = min(kmin,k)
				kmax = max(kmax,k)
				maxes = [min(maxes[0],j), max(maxes[1],j),min(maxes[2],k), max(maxes[3],k)]
				m+=1
				small = []
				if y[j-1][k-1]:
					y[j-1][k-1] = False
					small.append([j-1,k-1])
				if y[j][k-1]:
					y[j][k-1] = False
					small.append([j,k-1])
				if y[j+1][k-1]:
					y[j+1][k-1] = False
					small.append([j+1,k-1])
				if y[j-1][k]:
					y[j-1][k] = False
				small.append([j-1,k])
				if y[j+1][k]:
					y[j+1][k] = False
					small.append([j+1,k])
				if y[j-1][k+1]:
					y[j-1][k+1] = False
					small.append([j-1,k+1])
				if y[j][k+1]:
					y[j][k+1] = False
					small.append([j,k+1])
				if y[j+1][k+1]:
					y[j+1][k+1] = False
					small.append([j+1,k+1])
				if len(small)>1:
					for s in small: 
						q.put(s)
						temp.append(s)
			if len(ctemp)>5: c += ctemp 
		if m > max1[0]:
			max2 = max1
			max1=[m,temp,[jmin,jmax,kmin,kmax]]
		elif m > max2[0]:
			max2 = [m,temp, [jmin,jmax,kmin,kmax]]
		count +=m
	if max2[0] == 0: max2 = False
	if max1[0] == 0: max1 = False
	combined = [count,c, maxes] 
	return max1, max2, combined
def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L
def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])
def baseline_model():
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	sgd = SGD(lr=0.01, momentum=0.9,nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

def seperable(max1,max2,empty):
	t1 = numpy.empty_like (empty)
	t1[:] = empty
	for p in max1[1]:
		t1[p[0],p[1]] = 255 
	[jmin,jmax,kmin,kmax]=  resize(max1[2])
	t1 = t1[jmin:jmax,kmin:kmax]
	[jmin,jmax,kmin,kmax]=resize(max2[2])
	t2 = numpy.empty_like (empty)
	t2[:] = empty
	for p in max2[1]:
		t2[p[0],p[1]] = 255 
	
	t2 = t2[jmin:jmax,kmin:kmax]
	if t1.shape != (28,28): t1 = scipy.misc.imresize(t1,[28,28])
	if t2.shape != (28,28): t2= scipy.misc.imresize(t2,[28,28])
	
	return t1, t2
def Net(X,Y):
	
	K.set_image_dim_ordering('th')
	seed = 7
	numpy.random.seed(seed)

	X_train = numpy.array(X)
	y_train = numpy.array(Y)
	
	X_train = X_train.reshape(X_train.shape[0], 1, 60, 60).astype('float32')
	

	X_train = X_train / 255

	y_train = np_utils.to_categorical(y_train)
	

	model = VGG_16()
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')
	model.fit(X_train, y_train, nb_epoch=10, batch_size=200, verbose=2,validation_split=0.2)
	model.save('current.h5')

	

	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Baseline Error: %.2f%%" % (100-scores[1]*100))
def Classify(X):
	K.set_image_dim_ordering('th')
	seed = 7
	numpy.random.seed(seed)

	X_test1 = numpy.array(X[0])
	X_test2 = numpy.array(X[1])
	X_test1 = X_test1.reshape(X_test1.shape[0], 1, 28, 28).astype('float32')
	X_test2 = X_test2.reshape(X_test2.shape[0], 1, 28, 28).astype('float32')
	X_test1 = X_test1 / 255
	X_test2 = X_test2 / 255
	model = keras.models.load_model('3.2layers.h5')
	first =  model.predict_classes(X_test1, batch_size=16, verbose=0)
	second = model.predict_classes(X_test2, batch_size=16, verbose=0)
	Y = []
	for i in range(0,len(first)):
		Y.append([i,first[i]+second[i]])
	with open("test_out.csv",'wb') as l:
		writer=csv.writer(l,delimiter=',')
		writer.writerows( Y)	
def Nerual(X,Y):
	
	K.set_image_dim_ordering('th')
	seed = 7
	numpy.random.seed(seed)

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train= numpy.concatenate((X_train,X_test), axis = 0)
	y_train= numpy.concatenate((y_train,y_test), axis= 0 )
	X_test1 = numpy.array(X[0])
	X_test2 = numpy.array(X[1])

	y_test = Y
	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
	X_test1 = X_test1.reshape(X_test1.shape[0], 1, 28, 28).astype('float32')
	X_test2 = X_test2.reshape(X_test2.shape[0], 1, 28, 28).astype('float32')
	# temp = []
	# i = 0 
	# for x in X_train: 
	# 	i+=1
	# 	if i% 1000 == 0 : print i 
	# 	for j in range (0,28):
	# 		x[0][0][j] = 0 
	# 		x[0][j][0] = 0
	# 		x[0][27][j] = 0 
	# 		x[0][j][27] = 0 
	# 	for j in range(0,x[0].shape[0]):
	# 		for k in range(0,x[0].shape[1]):
	# 			if x[0][j][k] >=75:
	# 				box= 0 
	# 				b= 75
	# 				if x[0][j-1][k-1] >=b: box+=1
	# 				if x[0][j][k-1]  >=b: box+=1
	# 				if x[0][j+1][k-1]  >=b: box+=1
	# 				if x[0][j-1][k]  >=b: box+=1
	# 				if x[0][j+1][k]  >=b: box+=1
	# 				if x[0][j-1][k+1] >=b: box+=1
	# 				if x[0][j][k+1]  >=b: box+=1
	# 				if x[0][j+1][k+1]  >=b: box+=1
	# 				if box <= 1:
	# 					x[0][j][k] = 0 
	# 				else: 
	# 					x[0][j][k]=255
	# 			else:
	# 				x[0][j][k] = 0 
	# 	temp.append(x)
		

	# X_train = numpy.array(temp)
	X_train= pickle.load(open("training.p",'rb'))


	X_train = X_train / 255
	X_test1 = X_test1 / 255
	X_test2 = X_test2 / 255
	y_train = np_utils.to_categorical(y_train)
	# model = baseline_model()
	# model.fit(X_train, y_train, nb_epoch=3, batch_size=200, verbose=2, validation_split=.02)
	# model.save('1layers.h5')
	# model.fit(X_train, y_train, nb_epoch=3, batch_size=200, verbose=2, validation_split=.02)
	# model.save('2layers.h5')
	# model.fit(X_train, y_train, nb_epoch=3, batch_size=200, verbose=2, validation_split=.02)
	# model.save('3layers.h5')
	# model.fit(X_train, y_train, nb_epoch=3, batch_size=100, verbose=2)
	# model.save('4layers.h5')
	

	model = keras.models.load_model('4layers.h5')
	first =  model.predict_classes(X_test1, batch_size=16, verbose=0)
	cumul = []

	second = model.predict_classes(X_test2, batch_size=16, verbose=0)

	for i in range(0,len(first)):
		cumul.append(first[i]+second[i])
	
	error = 0.0
	ReturnX = [[],[]]
	ReturnY = []
	for i in range(0,len(cumul)):
		if cumul[i] != Y[i]: 
			# print " the guess was " + str(cumul[i]) + "the rel was " + str(Y[i])
			error+=1.0
			
			# print model.predict(numpy.array([X_test1[i]]), batch_size=1, verbose=0)
			# plt.imshow(X_test1[i][0])
			# plt.show()
			# print model.predict(numpy.array([X_test2[i]]), batch_size=1, verbose=0)
			# plt.imshow(X_test2[i][0])
			# plt.show()
		else: 
			ReturnX[0].append(X[0][i])
			ReturnX[1].append(X[1][i])
			ReturnY.append(Y[i])
	print " the error is " + str(error/float(len(cumul)))
	return ReturnX, ReturnY
	# scores = model.evaluate(X_test, y_test, verbose=0)
	# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
def FindSplit(params, empty, j,k, pa  ):
	m,dots,[jmin,jmax,kmin,kmax] = params 
	tries = []
	for i in range(0,16):
		e = numpy.empty_like (empty)
		e[:] =  empty 
		temp = [e]
		tries.append(temp)
	
	y1,x1 = [(jmax+jmin)/2 -pa+j*3, (kmax+kmin)/2 -pa +k*3]
	tries = numpy.array(tries)
	xm = [[0,60] for i in range(0,16)]
	ym=[[0,60] for i in range(0,16)]
	
	for dot in dots:
		y,x = dot
		if y <=y1:
			tries[0][0][dot[0]][dot[1]] = 255 
			xm[0][0] = max(x,xm[0][0])
			xm[0][1] = min(x,xm[0][1])
			ym[0][0] = max(y,ym[0][0])
			ym[0][1] = min(y,ym[0][1])
		else:
			 tries[1][0][dot[0]][dot[1]] = 255
			 xm[1][0] = max(x,xm[1][0])
			 xm[1][1] = min(x,xm[1][1])
			 ym[1][0] = max(y,ym[1][0])
			 ym[1][1] = min(y,ym[1][1])
		if y>= x/2 +y1-x1/2:
			 tries[2][0][dot[0]][dot[1]] = 255 
			 xm[2][0] = max(x,xm[2][0])
			 xm[2][1] = min(x,xm[2][1])
			 ym[2][0] = max(y,ym[2][0])
			 ym[2][1] = min(y,ym[2][1])
		else:
			 tries[3][0][dot[0]][dot[1]] = 255 
			 xm[3][0] = max(x,xm[3][0])
			 xm[3][1] = min(x,xm[3][1])
			 ym[3][0] = max(y,ym[3][0])
			 ym[3][1] = min(y,ym[3][1])
		if  y>= x*2 +y1-2*x1 :
			 tries[4][0][dot[0]][dot[1]] = 255
			 xm[4][0] = max(x,xm[4][0])
			 xm[4][1] = min(x,xm[4][1])
			 ym[4][0] = max(y,ym[4][0])
			 ym[4][1] = min(y,ym[4][1])
		else:
			 tries[5][0][dot[0]][dot[1]] = 255
			 xm[5][0] = max(x,xm[5][0])
			 xm[5][1] = min(x,xm[5][1])
			 ym[5][0] = max(y,ym[5][0])
			 ym[5][1] = min(y,ym[5][1])
		if x>=x1:
			 tries[6][0][dot[0]][dot[1]] = 255 
			 xm[6][0] = max(x,xm[6][0])
			 xm[6][1] = min(x,xm[6][1])
			 ym[6][0] = max(y,ym[6][0])
			 ym[6][1] = min(y,ym[6][1])
		else:
			 tries[7][0][dot[0]][dot[1]] = 255 
			 xm[7][0] = max(x,xm[7][0])
			 xm[7][1] = min(x,xm[7][1])
			 ym[7][0] = max(y,ym[7][0])
			 ym[7][1] = min(y,ym[7][1])
		if y<= x*-2 +y1+x1*2:
			 tries[8][0][dot[0]][dot[1]] = 255 
			 xm[8][0] = max(x,xm[8][0])
			 xm[8][1] = min(x,xm[8][1])
			 ym[8][0] = max(y,ym[8][0])
			 ym[8][1] = min(y,ym[8][1])
		else:
			 tries[9][0][dot[0]][dot[1]] =255
			 xm[9][0] = max(x,xm[9][0])
			 xm[9][1] = min(x,xm[9][1])
			 ym[9][0] = max(y,ym[9][0])
			 ym[9][1] = min(y,ym[9][1])
		if  y<= x/-2 +y1+x1/2:
			 tries[10][0][dot[0]][dot[1]] = 255 
			 xm[10][0] = max(x,xm[10][0])
			 xm[10][1] = min(x,xm[10][1])
			 ym[10][0] = max(y,ym[10][0])
			 ym[10][1] = min(y,ym[10][1])
		else:
			 tries[11][0][dot[0]][dot[1]] = 255
			 xm[11][0] = max(x,xm[11][0])
			 xm[11][1] = min(x,xm[11][1])
			 ym[11][0] = max(y,ym[11][0])
			 ym[11][1] = min(y,ym[11][1])
		if y>= x +y1-x1:
			 tries[12][0][dot[0]][dot[1]] = 255
			 xm[12][0] = max(x,xm[12][0])
			 xm[12][1] = min(x,xm[12][1])
			 ym[12][0] = max(y,ym[12][0])
			 ym[12][1] = min(y,ym[12][1])
		else:
			 tries[13][0][dot[0]][dot[1]] = 255
			 xm[13][0] = max(x,xm[13][0])
			 xm[13][1] = min(x,xm[13][1])
			 ym[13][0] = max(y,ym[13][0])
			 ym[13][1] = min(y,ym[13][1])
		if  y<= x +y1+x1:
			 tries[14][0][dot[0]][dot[1]] = 255 
			 xm[14][0] = max(x,xm[14][0])
			 xm[14][1] = min(x,xm[14][1])
			 ym[14][0] = max(y,ym[14][0])
			 ym[14][1] = min(y,ym[14][1])
		else: 
			 tries[15][0][dot[0]][dot[1]] = 255 
			 xm[15][0] = max(x,xm[15][0])
			 xm[15][1] = min(x,xm[15][1])
			 ym[15][0] = max(y,ym[15][0])
			 ym[15][1] = min(y,ym[15][1])

	Final = []
	for i in range (0,16):
		jmin,jmax,kmin,kmax= resize([ym[i][1],ym[i][0], xm[i][1],xm[i][0]])
		temp = tries[i][0][jmin:jmax,kmin:kmax]
		if temp.shape != (28,28): temp = scipy.misc.imresize(temp,[28,28])
		Final.append([temp])
	Final = numpy.array(Final)
	Final.reshape(Final.shape[0], 1, 28, 28).astype('float32')
	Final= Final / 255

	pred =  list(model.predict(Final, batch_size=16, verbose=0))
	maxset  = []
	for i in range(0,len(Final)/2):
		maxset.append(max(pred[2*i])+max(pred[2*i+1]))
	ind= maxset.index(max(maxset))
	# ri = "the prediction is " + str( list(pred[2*ind]).index(max(pred[2*ind])) + list(pred[2*ind+1]).index(max(pred[2*ind+1]))) 
	return Final[ind*2][0]*255, Final[ind*2+1][0]*255, max(maxset), [max(pred[2*ind]), max(pred[2*ind+1]) ]
def FindBest(max1, empty, x ):
	
	output = Try(max1, empty,5, 6)

	# if output[3][0]>0.9999 and output[3][1]>0.9999:
	return output
	d = numpy.empty_like (x)
	d[:] = x
	for j in range (0,60):
			x[0][j] = 0 
			x[j][0] = 0
			x[59][j] = 0 
			x[j][59] = 0 	
	points = []
	y= [[False for t in range(0, 60)] for j in range(0,60)]
	for j in range(1,60):
		for k in range( 1,60):
			if x[j][k] >=255:
				box= 0 
				b = 245
				if x[j-1][k-1] >=b: box+=1
				if x[j][k-1]  >=b: box+=1
				if x[j+1][k-1]  >=b: box+=1
				if x[j-1][k]  >=b: box+=1
				if x[j+1][k]  >=b: box+=1
				if x[j-1][k+1] >=b: box+=1
				if x[j][k+1]  >=b: box+=1
				if x[j+1][k+1]  >=b: box+=1
				if box >=2:
					points.append([j,k])
					y[j][k] = True
					x[j][k]=255
	max1, max2 , combined= top2( points, y)
	output2 = Try(max1,empty,3, 3)
	if output2[3][0]>0.9999 and output2[3][1]>0.9999:
		return output2

	for j in range (0,60):
			d[0][j] = 0 
			d[j][0] = 0
			d[59][j] = 0 
			d[j][59] = 0 	
	points = []
	y= [[False for t in range(0, 60)] for j in range(0,60)]
	for j in range(1,60):
		for k in range( 1,60):
			if d[j][k] >=250:
				box= 0 
				b = 200
				if d[j-1][k-1] >=b: box+=1
				if d[j][k-1]  >=b: box+=1
				if d[j+1][k-1]  >=b: box+=1
				if d[j-1][k]  >=b: box+=1
				if d[j+1][k]  >=b: box+=1
				if d[j-1][k+1] >=b: box+=1
				if d[j][k+1]  >=b: box+=1
				if d[j+1][k+1]  >=b: box+=1
				if box >=1:
					points.append([j,k])
					y[j][k] = True
					d[j][k]=255
	max1, max2 , combined= top2( points, y)
	output3 = Try(max1,empty,3, 3)

	if output3[3][0]>0.9999 and output3[3][1]>0.9999:
		return output3
	k = max(output[3][0] + output[3][1], output2[3][0] + output2[3][1], output3[3][0] + output3[3][1] )
	if output[3][0] + output[3][1] == k : return output
	if output[3][0] + output[3][1] == k : return output2
	else: return output3
def Try(max1, empty,i, pa):
	m = [0,0,0,0]
	for j in range(0,i):
		for k in range(0,i):
			t1,t2,pre,pred = FindSplit(max1, empty,j,k, pa)
			if pre>m[2]: m = [t1,t2,pre,pred]
	return m 
def PreProcess():
	x = numpy.fromfile('train_x.bin', dtype='uint8')
	x = x.reshape((100000,60,60))
	fh = open('train_y.csv')
	csvY = csv.reader(fh)
	track=0
	Ytemp = []
	for row in csvY:
		if track> len(x):
			break
		Ytemp.append(int(row[1]))
	empty = numpy.empty_like (x[0])
	for i in range(0,60):
		for j in range(0,60):
			empty[i][j] = 0 
	error= 0
	Y=[]
	X=[[],[]]
	for i in range(30000,45000):
		skip = True 
		if i % 100 == 0 and i>0: print i 
		y= [[False for t in range(0, 60)] for j in range(0,60)]
		copy = numpy.empty_like (x[i])
		
		for j in range (0,60):
			x[i][0][j] = 0 
			x[i][j][0] = 0
			x[i][59][j] = 0 
			x[i][j][59] = 0 
		copy[:] = x[i]
		points = []
		ruts = []
		for j in range(1,60):
			for k in range( 1,60):
				if x[i][j][k] >=255:
					box= 0 
					b = 245
					if x[i][j-2][k-1] >=b: box+=1
					if x[i][j-1][k-1] >=b: box+=1
					if x[i][j][k-1]  >=b: box+=1
					if x[i][j+1][k-1]  >=b: box+=1
					if x[i][j-1][k]  >=b: box+=1
					if x[i][j+1][k]  >=b: box+=1
					if x[i][j-1][k+1] >=b: box+=1
					if x[i][j][k+1]  >=b: box+=1
					if x[i][j+1][k+1]  >=b: box+=1

					if box >=1:
						points.append([j,k])
						y[j][k] = True
						x[i][j][k]=255
					else: ruts.append([j,k])
				elif x[i][j][k] >=245: ruts.append([j,k])
				else: x[i][j][k]= 0 
		for [j,k] in ruts: 
			x[i][j][k] = 0 


		# U,L= hulls(points)
		hull = points
		max1, max2 , combined= top2( hull, y)
		if not max1:
			print "this one sucks, but im not a rapper" 
			
		elif not max2: 
			continue		
			t1, t2 , m, prediction = FindBest(combined,empty, copy)
			Y.append(Ytemp[i])
		else:  
			if float(max2[0])/float(combined[0])<.3:
				continue
				t1, t2 , m, prediction = FindBest(combined,empty,copy)
			else: 
				t1,t2= seperable(max1,max2,empty)
			Y.append(Ytemp[i])



		X[0].append(t1)
		X[1].append(t2)	
	return X,numpy.array(Y)
def PreProcessBoarder():
	x = numpy.fromfile('train_x.bin', dtype='uint8')
	x = x.reshape((10000,60,60))
	fh = open('train_y.csv')
	csvY = csv.reader(fh)
	track=0
	Ytemp = []
	for row in csvY:
		if track> len(x):
			break
		Ytemp.append(int(row[1]))
	empty = numpy.empty_like (x[0])
	for i in range(0,60):
		for j in range(0,60):
			empty[i][j] = 0 
	mpty = numpy.empty_like (x[0])
	for i in range(0,60):
		for j in range(0,60):
			mpty[i][j] = 0 
	error= 0
	Y=[]
	X=[[],[]]
	for i in range(0,len(x)):
		 
		if i % 20 == 0 and i>0: print i 
		y= [[False for t in range(0, 60)] for j in range(0,60)]
		copy = numpy.empty_like (x[i])
		
		for j in range (0,60):
			x[i][0][j] = 0 
			x[i][j][0] = 0
			x[i][59][j] = 0 
			x[i][j][59] = 0 
		copy[:] = x[i]
		points = []
		boarder= []
		for j in range(1,60):
			for k in range( 1,60):
				if x[i][j][k] >=255:
					temper= []
					box= 0 
					b = 245
					if x[i][j-1][k-1] >=b: 
						box+=1
						if not([j-1,k-1] in boarder) and x[i][j-1][k-1]!= 255:
							temper.append([j-1,k-1], x[i][j-1][k-1])
					if x[i][j][k-1]  >=b: 
						box+=1
						if not([j,k-1] in boarder) and x[i][j][k-1]!= 255:
							temper.append([j,k-1], x[i][j][k-1])
					if x[i][j+1][k-1]  >=b: 
						box+=1
						if not([j+1,k-1] in boarder) and x[i][j+1][k-1]!= 255:
							temper.append([j+1,k-1],  x[i][j+1][k-1])
					if x[i][j-1][k]  >=b: 
						box+=1
						if not([j-1,k] in boarder) and x[i][j-1][k]!= 255:
							temper.append([j-1,k], x[i][j-1][k])
					if x[i][j+1][k]  >=b: 
						box+=1
						if not([j+1,k] in boarder) and x[i][j+1][k]!= 255:
							temper.append([j+1,k], x[i][j+1][k])
					if x[i][j-1][k+1] >=b: 
						box+=1
						if not([j-1,k+1] in boarder) and x[i][j-1][k+1]!= 255:
							temper.append([j-1,k+1], x[i][j-1][k+1])
					if x[i][j][k+1]  >=b:
						box+=1
						if not([j,k+1] in boarder) and x[i][j][k+1]!= 255:
							temper.append([j,k+1], x[i][j][k+1])
					if x[i][j+1][k+1]  >=b: 
						box+=1
						if not([j+1,k+1] in boarder) and x[i][j+1][k+1]!= 255:
							temper.append([j+1,k+1], x[i][j+1][k+1])
					if box >=1:
						boarder+= temper 
						points.append([j,k])
						y[j][k] = True
						x[i][j][k]=255
		for [[j,k], col] in boarder:
			mpty[j][k] = col  
		# U,L= hulls(points)
		hull = points
		max1, max2 , combined= top2( hull, y)
		if not max1:
			print "this one sucks, but im not a rapper" 
			continue
		elif not max2: 
			
			t1, t2 , m, prediction = FindBest(combined,empty, copy)
			Y.append(Ytemp[i])
		else:  
			if float(max2[0])/float(combined[0])<.25:
				
				t1, t2 , m, prediction = FindBest(combined,empty,copy)
			else: 
				t1,t2= seperable(max1,max2,empty)
			Y.append(Ytemp[i])
	
		X[0].append(t1)
		X[1].append(t2)	
	return X,numpy.array(Y)
def Process():

	x = numpy.fromfile('train_x.bin', dtype='uint8')
	x = x.reshape((100000,60,60))
	fh = open('train_y.csv')
	csvY = csv.reader(fh)
	Y=[]
	X=[]
	for row in csvY:
		Y.append(int(row[1]))
	for i in range(0,len(x)):
		if i % 100 == 0 and i>0: print i 
			
		temp = numpy.empty_like (x[i])
		temp[:] = x[i]
		for j in range(1,59):
			for k in range( 1,59):
				if x[i][j][k] >=250:
					box= 0 
					if x[i][j-1][k-1] >=250: box+=1
					if x[i][j][k-1]  >=250: box+=1
					if x[i][j+1][k-1]  >=250: box+=1
					if x[i][j-1][k]  >=250: box+=1
					if x[i][j+1][k]  >=250: box+=1
					if x[i][j-1][k+1] >=250: box+=1
					if x[i][j][k+1]  >=250: box+=1
					if x[i][j+1][k+1]  >=250: box+=1
					if box <= 1:
						temp[j][k] = 0 
					else: 
						temp[j][k]=255
				else:
					temp[j][k] = 0
		for j in range (0,60):
			temp[0][j] = 0
			temp[j][0] = 0
			temp[59][j] = 0
			temp[j][59] = 0 
		X.append(temp)

	return X,Y
def Logistic(X,Y):
	Prime = numpy.empty([len(X[0]),28,28])
	Prime2 = numpy.empty([len(X[0]),28,28])
	print len(X[0])
	for i,x in  enumerate(X[0]):
		Prime[i][:] = x

	for i,x in enumerate(X[1]):
		Prime2[i][:] = x
	X1= numpy.empty([len(Prime),28*28])
	for i,p in enumerate(Prime):
		X1[i] = p.flatten()
	X2= numpy.empty([len(Prime2),28*28])
	for i,p in enumerate(Prime2):
		X2[i] = p.flatten()
	


	logistic = linear_model.LogisticRegression()

 	j = 80
	pca = decomposition.PCA(n_components=j)
	pca.fit(X1)
	X1= pca.transform(X1)

	pca = decomposition.PCA(n_components=j)
	pca.fit(X2)
	X2= pca.transform(X2)
	X = numpy.concatenate((X1, X2), axis=0)
	est= LogisticRegression(C=1.0, penalty='l2', tol=0.01)

	est.fit(X[:16000], Y[:16000])
	acc = 0.0
	for i in range(16000,18351):
		if est.predict([X[i]]) == Y[i]: acc +=1.0
	print "the accuracy for" +  str(j)+ " is : " + str(acc/(18351.0-16000.0))
		
def Augment(X,Y):
	final = []
	for i in range(len(X[0])):
		while True:
			f = numpy.zeros((60,60))
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			for j in range(0,28):
				for k in range(0, 28):
					if X[0][i][j][k] == 255:
						f[r1+j][r2+k] = 255
			c = False
			rate = 0 
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			g = numpy.zeros((60,60))
			g[:] = f
			for j in range(0,28):
				for k in range(0, 28):
					if f[r1+j][r2+k] == 255: 
						rate+=1
						if rate> 5:
							c = True 
					g[r1+j][r2+k] = X[1][i][j][k]
				if c == True: break 
			if c == False: break 
		final.append(g)
		while True:
			f = numpy.zeros((60,60))
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			for j in range(0,28):
				for k in range(0, 28):
					if X[0][i][j][k] == 255:
						f[r1+j][r2+k] = 255
			c = False
			rate = 0 
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			g = numpy.zeros((60,60))
			g[:] = f
			for j in range(0,28):
				for k in range(0, 28):
					if f[r1+j][r2+k] == 255: 
						rate+=1
						if rate> 5:
							c = True 
					g[r1+j][r2+k] = X[1][i][j][k]
				if c == True: break 
			if c == False: break 
		final.append(g)
		while True:
			f = numpy.zeros((60,60))
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			for j in range(0,28):
				for k in range(0, 28):
					if X[0][i][j][k] == 255:
						f[r1+j][r2+k] = 255
			c = False
			rate = 0 
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			g = numpy.zeros((60,60))
			g[:] = f
			for j in range(0,28):
				for k in range(0, 28):
					if f[r1+j][r2+k] == 255: 
						rate+=1
						if rate> 5:
							c = True 
					g[r1+j][r2+k] = X[1][i][j][k]
				if c == True: break 
			if c == False: break 
		final.append(g)
		while True:
			f = numpy.zeros((60,60))
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			for j in range(0,28):
				for k in range(0, 28):
					if X[0][i][j][k] == 255:
						f[r1+j][r2+k] = 255
			c = False
			rate = 0 
			r1 = random.randrange(0, 60-28, 2)
			r2 = random. randrange(0, 60-28, 2)
			g = numpy.zeros((60,60))
			g[:] = f
			for j in range(0,28):
				for k in range(0, 28):
					if f[r1+j][r2+k] == 255: 
						rate+=1
						if rate> 5:
							c = True 
					g[r1+j][r2+k] = X[1][i][j][k]
				if c == True: break 
			if c == False: break 
		final.append(g)
	numpy.save('Aug2X',numpy.array(final))
	Y2 = []
	for y in Y:
		Y2.append(y)
		Y2.append(y)
		Y2.append(y)
		Y2.append(y)
	numpy.save('Aug2Y',numpy.array(Y))









####################################################################### 
# X,Y = PreProcess()
# # pickle.dump(X,open('XBest.p','wb'))
# # pickle.dump(Y,open('Ybext.p', 'wb'))
# numpy.save('secondGoX',numpy.array(X))
# numpy.save('secondGoY',numpy.array(Y))
# # for x in X:
# # 	plt.imshow(x)
# # 	plt.show()
# #Logistic(X,Y)
# # plt.imshow(X[0][1])
# # plt.show()


# # Classify(X)
# # GoodX, GoodY = Nerual(X,Y)
# # pickle.dump(GoodX,open('GoodX.p','wb'))
# # pickle.dump(GoodY,open('GoodY.p', 'wb'))
# # X,Y = Nerual(X,Y)
# Augment(X,Y)



Y = numpy.load('Aug2Y.npy')
Y2 = []
for y in Y:
	Y2.append(y)
	Y2.append(y)
	Y2.append(y)
	Y2.append(y)
numpy.save('Aug2Y',numpy.array(Y2))



