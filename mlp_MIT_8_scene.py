from array import array
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
from trainings import net_configurations, test_name

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

if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()


print('Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = createModel()

print('Done!\n')

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')


# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)




batch_sizes = [8, 16, 64, 128]
img_sizes = [32, 64, 128, 256]


for size in img_sizes:
  for batch in batch_sizes:
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generatorAux = train_datagen.flow_from_directory(
            DATASET_DIR+'/train',  # this is the target directory
            target_size=(size, size),  # all images will be resized to sizexsize
            batch_size=batch,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generatorAux = test_datagen.flow_from_directory(
            DATASET_DIR+'/test',
            target_size=(size, size),
            batch_size=batch,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')

    first = ['Reshape', size, size]


    for name, layers in zip(test_name, net_configurations):
      layers[0] = first
      print(layers)
      save_name = 'results/batch_' + str(batch) + '-sz_' + str(size) + name
      print('Starting new NN test:', save_name)
      
      modelAux = createModelByLayersSimple(layers)
      plot_model(modelAux, to_file=(save_name+'_model.png'), show_shapes=True, show_layer_names=True)
      historyAux = modelAux.fit_generator(
            train_generatorAux,
            steps_per_epoch=1881 // batch,
            epochs=50,
            validation_data=validation_generatorAux,
            validation_steps=807 // batch,
            verbose=0)
      saveModel(modelAux, historyAux, save_name)


# history = model.fit_generator(
#         train_generator,
#         steps_per_epoch=1881 // BATCH_SIZE,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=807 // BATCH_SIZE,
#         verbose=1)

# print('Done!\n')
# print('Saving the model into '+MODEL_FNAME+' \n')
# model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
# print('Done!\n')
#   # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.savefig('accuracy.jpg')
# plt.close()
#   # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.savefig('loss.jpg')

#to get the output of a given layer
 #crop the model up to a certain layer
# model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

# #get the features from images
# directory = DATASET_DIR+'/test/coast'
# x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
# x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
# print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0] ))
# features = model_layer.predict(x/255.0)
# print(features)
# print('Done!')
