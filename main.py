#main
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
from model import runCNN

global filename
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test
global cnn
global labels

labels = ['No_DR','Mild', 'Moderate','Proliferate_DR','Severe']
medicine = ['None','Eye drops', 'Laser therapy', 'anti-VEGF injections','Surgical Intervention']
lifestyle = [['Eat healthy with lots of fruits and veggies', 'Exercise regularly and stay active', 'Wear sunglasses to shield your eyes from sunlight'],
             ['Eat heart-healthy foods', 'Keep an eye on your blood sugar levels', 'Exercise to help manage diabetes', 'Quit smoking'],
             ['Eat foods that are good for your heart', 'Use prescribed eye drops and medications', 'Exercise moderately', 'Check your eye pressure regularly'],
             ['Take your medications and eye drops as prescribed', 'Manage stress to avoid eye pressure spikes', 'Do gentle exercises to stay healthy', 'Limit caffeine intake']
             ,['Follow a low-sodium diet', 'Stick to your treatment plan diligently', 'Get emotional support as needed', 'Work closely with your healthcare team for aggressive management']]



main = tkinter.Tk()
main.title("Glaucoma Disease detection and Classification using ML and DL-TL") #designing main screen
main.geometry("1300x1200")

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False): #extract glcm features
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = graycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image): #texture features will be extracted using histogram
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    graycoprops(glcm, 'correlation'),
    graycoprops(glcm, 'contrast'),
    graycoprops(glcm, 'energy'),
    graycoprops(glcm, 'homogeneity'),
    graycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    text.insert(END,"Different Diseases Found in Dataset : "+str(labels)+"\n\n")
    text.insert(END,"Total diseases are : "+str(len(labels)))

def featuresExtraction():
    global filename
    path="Dataset"
    global X,Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    class_label = getID(name)
                    features = extract_features(img)
                    Y.append(class_label)
                    X.append(features)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)

    X = X.astype('float32')
    X = X/255

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Extracted GLCM & Texture Features : "+str(X[0])+"\n\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split. 80% dataset images used for training and 20% for testing\n\n")
    text.insert(END,"80% training images : "+str(X_train.shape[0])+"\n\n")
    text.insert(END,"20% training images : "+str(X_test.shape[0])+"\n\n")

def calculateMetrics():
    global cnn  
    algorithm, predict, Y_test, cnn = runCNN(X_train, X_test, y_train, y_test, X, Y)
    global accuracy, precision,recall, fscore
    a = accuracy_score(Y_test,predict)*100
    p = precision_score(Y_test, predict,average='macro') * 100
    r = recall_score(Y_test, predict,average='macro') * 100
    f = f1_score(Y_test, predict,average='macro') * 100
    print(a)
    print(p)
    print(r)
    print(f)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" F1core    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(Y_test, predict)
    plt.figure(figsize =(6, 3))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix")
    plt.xticks(rotation=90)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()





def graph():
    df = pd.DataFrame([['Propose CNN-TL','Accuracy',accuracy[0]],['Propose CNN-TL','Precision',precision[0]],['Propose CNN-TL','Recall',recall[0]],['Propose CNN-TL','FScore',fscore[0]],

                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predict():
    global cnn
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    test = []
    img = cv2.resize(img, (64,64))
    features = extract_features(img)
    test.append(features)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = cnn.predict(test)
    predict = np.argmax(predict)

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, 'Retina Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Medicine Recommended is : '+medicine[predict], (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Lifestyle Recommendations are : ', (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    a=85
    for i in range(0,len(lifestyle[predict])):
       cv2.putText(img,'    ->'+lifestyle[predict][i], (10, a),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
       a+=20
    cv2.imshow('Retina Predicted as : '+labels[predict], img)

    cv2.waitKey(0)

font = ('times', 16, 'bold')
title = Label(main, text='Glaucoma detection using Deep Learning ')
title.config(bg='greenyellow', fg='dodger blue')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Kaggle Dataset", command=uploadDataset)
uploadButton.place(x=50,y=500)
uploadButton.config(font=font1)

featuresButton = Button(main, text="Extract Texture & GLCM Features", command=featuresExtraction)
featuresButton.place(x=370,y=500)
featuresButton.config(font=font1)

cnnButton = Button(main, text="Run Propose CNN-TL Algorithm", command=calculateMetrics)
# calculateMetrics("Propose CNN", predict, Y_test)
cnnButton.place(x=750,y=500)
cnnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=550)
graphButton.config(font=font1)


predictButton = Button(main, text="Disease Detection from Test Image", command=predict)
predictButton.place(x=370,y=550)
predictButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()