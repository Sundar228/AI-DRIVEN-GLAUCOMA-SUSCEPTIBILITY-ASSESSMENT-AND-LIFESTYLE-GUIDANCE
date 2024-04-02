#model

from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pickle

import cv2
import skimage
import tensorflow as tf
import pydot
from skimage import color
from skimage.feature import graycomatrix, graycoprops
import scipy.stats as stats
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.applications import VGG16
from keras.utils.vis_utils import plot_model








def runCNN(X_train, X_test, y_train, y_test, X, Y):
    
    

    Y1 = to_categorical(Y)
    gabor_china_X_train = np.repeat(X, 3, axis=-1)
    height=648
    width=648
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y1, test_size=0.2)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        json_file.close()
        cnn.load_weights("model/model_weights.h5")
        cnn._make_predict_function()

        
    else:

        input_shap = (height, width, 3) 
        weights_path = 'model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        vgg16_model = VGG16(weights=weights_path, include_top=False, input_shape=input_shap)
        cnn = Sequential()
        cnn.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1], XX.shape[2], XX.shape[3]), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim = 256, activation = 'relu'))
        cnn.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # for compiling the model
        
        hist = cnn.fit(XX, Y1, batch_size=12, epochs=100, shuffle=True, verbose=2) #for Training the model
        cnn.save_weights('model/model_weights.h5')
        model_json = cnn.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn.summary())
    # plot_model(cnn, to_file='model/model_diagram.png', show_shapes=True, show_layer_names=True)

    predict = cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    return "Propose CNN", predict, y_test, cnn