import os
import getpass
import argparse


from utils import *

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.python.client import device_lib
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the task 1 script')
    parser.add_argument('-i', '--img_size', type=int, default=16, help='Image size')
    parser.add_argument('-m', '--model', default='model_trained.h5', type=str, help='Relative path from /home/group05/m3/ to store the model')
    parser.add_argument('-d', '--dataset', default='/home/mcv/datasets/MIT_split', type=str, help='Absolute path to the image dataset')

    return parser.parse_args()

args = parse_args()

#user defined variables
IMG_SIZE    = args.img_size
BATCH_SIZE  = 16
DATASET_DIR =  args.dataset
MODEL_FNAME =  '' + args.model

def createModel(activation='relu', neurons=2048):
#Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
    model.add(Dense(units=neurons, activation=activation,name='second'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    return model

if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

model = KerasClassifier(build_fn=createModel, epochs=50, batch_size=16, verbose=1)

neurons = [256, 1024, 2048]
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid']
param_grid = dict(neurons=neurons,activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)

train_images_filenames = pickle.load(open('train_images_filenames.dat','rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat','rb'))
train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
test_images_filenames  = ['..' + n[15:] for n in test_images_filenames]
train_labels = pickle.load(open('train_labels.dat','rb'))
test_labels = pickle.load(open('test_labels.dat','rb'))

print('Starting Grid training...')
train_images = []
for img in train_images_filenames:
    ima = cv2.imread(img)
    color = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
    train_images.append(color)

history = grid.fit(train_images, train_labels)

# summarize results
print("Best: %f using %s" % (history.best_score_, history.best_params_))
means = history.cv_results_['mean_test_score']
stds = history.cv_results_['std_test_score']
params = history.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))