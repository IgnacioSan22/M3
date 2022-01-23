from utils import *
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator

import pickle
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def get_descriptors(model, images_filenames):
    descriptors = np.empty((len(images_filenames), NUM_PATCHES, model.layers[-1].output_shape[1]))
    for i,filename in enumerate(images_filenames):
        img = Image.open(filename)
        patches = image.extract_patches_2d(np.array(img), PATCH_SIZE, max_patches=NUM_PATCHES)
        descriptors[i, :, :] = model.predict(patches/255.)

    return descriptors

def get_visual_words(descriptors, codebook, codebook_size):
    visual_words=np.empty((len(descriptors),codebook_size),dtype=np.float32)
    for i,des in enumerate(descriptors):
        words=codebook.predict(des)
        visual_words[i,:]=np.bincount(words,minlength=codebook_size)

    return StandardScaler().fit_transform(visual_words)

def build_mlp(input_size=64,phase='TRAIN'):
  model = Sequential()
  model.add(Reshape((input_size*input_size*3,),input_shape=(input_size, input_size, 3)))
  model.add(Dense(units=1024, activation='relu'))
  model.add(Dense(units=1024, activation='relu'))
  if phase=='TEST':
    model.add(Dense(units=8, activation='linear')) # In test phase we softmax the average output over the image patches
  else:
    model.add(Dense(units=8, activation='softmax'))
  return model

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = "/home/group05/m3/datasets_brian/MIT_split"
PATCHES_DIR = '/home/group05/m3/data/MIT_split_patches_64'
MODEL_FNAME = '/home/group05/m3/patch_based_mlp_64.h5'
# RESULTS_DIR = '/home/group05/m3/results_bow/'

if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

model = build_mlp(input_size=64)
model.load_weights(MODEL_FNAME)
model = Model(inputs=model.input, outputs=model.layers[-2].output)
model.summary()

PATCH_SIZE = model.layers[0].input.shape[1:3]
NUM_PATCHES = (IMG_SIZE//PATCH_SIZE.as_list()[0])**2

codebook_size = 512

train_descriptors = get_descriptors(model, train_images_filenames)

codebook = MiniBatchKMeans(n_clusters=codebook_size,
                            verbose=False,
                            batch_size=codebook_size * 20,
                            compute_labels=False,
                            reassignment_ratio=10**-4,
                            random_state=42)

codebook.fit(np.vstack(train_descriptors))

train_visual_words = get_visual_words(train_descriptors, codebook, codebook_size)

test_descriptors = get_descriptors(model, test_images_filenames)
test_visual_words = get_visual_words(test_descriptors, codebook, codebook_size)

classifier = svm.SVC(kernel='rbf')
classifier.fit(train_visual_words,train_labels)

accuracy = classifier.score(test_visual_words, test_labels)

print(f'Test accuracy: {accuracy}')

# compute_roc(train_visual_words, test_visual_words, train_labels, test_labels, classifier, RESULTS_DIR+'ROC.png')
# save_confusion_matrix(test_labels, classifier.predict(test_visual_words), RESULTS_DIR+'confusion_matrix.png')