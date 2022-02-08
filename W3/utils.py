from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Color:
    GRAY=30
    RED=31
    GREEN=32
    YELLOW=33
    BLUE=34
    MAGENTA=35
    CYAN=36
    WHITE=37
    CRIMSON=38    

def colorize(num, string, bold=False, highlight = False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))

def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size),max_patches=(int(np.asarray(im).shape[0]/patch_size)**2))#max_patches=1.0
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')


def createModel(activation='relu', neurons=2048, img_size = 32):
#Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((img_size*img_size*3,),input_shape=(img_size, img_size, 3),name='first'))
    model.add(Dense(units=neurons, activation=activation,name='second'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    return model

def createModelByLayers(layers):
#Build the Multi Layer Perceptron model
    model = Sequential()
    for layer in layers:
      model.add(layer)

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    return model

def createModelByLayersSimple(layers):
#Build the Multi Layer Perceptron model
    model = Sequential()
    for layer in layers:
      if layer[0] == 'Reshape':
        model.add(Reshape((layer[1]*layer[2]*3,),input_shape=(layer[1], layer[2], 3)))
      else:
        model.add(Dense(units=layer[1], activation=layer[2]))

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    return model

def saveModel(model, histroy, path):
  plt.plot(histroy.history['accuracy'])
  plt.plot(histroy.history['val_accuracy'])
  plt.title(f'model accuracy, layers: {len(model.layers)}')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.savefig(path+'_accuracy.jpg')
  plt.close()
  print('Saving the model into '+path+' \n')
  model.save_weights(path+'.h5')  # always save your weights after training or during training
  print('Done!\n')