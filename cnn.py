import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency

import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_bin_data(input):
    Z = np.fromfile(input, dtype='uint8')
    Z = (Z == 255) # binarize pixels
    Z = Z.reshape((-1,1,60,60)) # shape images
    return Z


def read_csv_data(input):
    with open(input, 'rt') as f:
        reader = csv.reader(f)
        lines = list(reader)[1:] # get rid of first line
    output = []
    for row in lines:
        output.append(row[1]) # take second element per row
    return output


"""prints the output to a csv file
   output_file : the path of the csv file to be written
   y : numpy vector of outputs
   """
def write_predictions(output_file, y):
    with open(output_file, 'wt') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id','Prediction'])
        i = 0
        for val in y:
            writer.writerow([i,val])
            i += 1


"""shows the first elements of X matrix
   and corresponding values in y"""
def show_some_data(X, y, num_images=3):
    for i in range(num_images):
        plt.imshow(X[i][0], cmap=plt.cm.binary)
        plt.show()
        print(y[i])


# read training data data
X = read_bin_data('data/train_x.bin')
y = np.array(read_csv_data('data/train_y.csv')).astype(np.uint8)

# read kaggle data
X_kaggle = read_bin_data('data/test_x.bin')


# Define our model
layers=[('input', InputLayer),
        ('conv2d1', Conv2DLayer),
        ('conv2d2', Conv2DLayer),
        ('maxpool1', MaxPool2DLayer),
        ('conv2d3', Conv2DLayer),
        ('conv2d4', Conv2DLayer),
        ('maxpool2', MaxPool2DLayer),
        ('conv2d5', Conv2DLayer),
        ('conv2d6', Conv2DLayer),
        ('maxpool3', MaxPool2DLayer),
        ('dropout1', DropoutLayer),
        ('dense', DenseLayer),
        ('dropout2', DropoutLayer),
        ('output', DenseLayer)]

net1 = NeuralNet(
        layers=layers,
        # input layer
        input_shape=(None, 1, 60, 60),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(3, 3),
        conv2d1_nonlinearity=rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),  
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(3, 3),
        conv2d2_nonlinearity=rectify,
        # layer maxpool1
        maxpool1_pool_size=(2, 2),    
        # layer conv2d3
        conv2d3_num_filters=32,
        conv2d3_filter_size=(3, 3),
        conv2d3_nonlinearity=rectify,
        # layer conv2d4
        conv2d4_num_filters=32,
        conv2d4_filter_size=(3, 3),
        conv2d4_nonlinearity=rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # layer conv2d5
        conv2d5_num_filters=32,
        conv2d5_filter_size=(3, 3),
        conv2d5_nonlinearity=rectify,
        # layer conv2d6
        conv2d6_num_filters=32,
        conv2d6_filter_size=(3, 3),
        conv2d6_nonlinearity=rectify,
        # layer maxpool3
        maxpool3_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units= 256,
        dense_nonlinearity=rectify,    
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=softmax,
        output_num_units=19,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.001,
        train_split=TrainSplit(eval_size=0.20),
        update_momentum=0.9,
        #objective_l2=1e-2,
        max_epochs=1,
        verbose=1,
        )


# Retrieve weight from saved model (optional)
#net1.load_params_from('models/model95')

# Train our CNN and
epochs = 30
for epoch in range(epochs):
    net1.fit(X, y)
    # Save parameters after every training epoch (optional)
    #net1.save_params_to('models/model9' + str(epoch+1))




