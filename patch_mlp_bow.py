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

train_images_filenames = pickle.load(open('train_images_filenames.dat','rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat','rb'))
train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
test_images_filenames  = ['..' + n[15:] for n in test_images_filenames]
train_labels = pickle.load(open('train_labels.dat','rb'))
test_labels = pickle.load(open('test_labels.dat','rb'))

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = "/content/drive/MyDrive/M3_Project/MIT_split"
PATCHES_DIR = '/content/drive/MyDrive/M3_Project/DeepLearning/MIT_split_patches'
MODEL_FNAME = '/content/drive/MyDrive/M3_Project/DeepLearning/patch_based_mlp.h5'
RESULTS_DIR = '/content/drive/MyDrive/M3_Project/DeepLearning/results/bow/'

if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

model = load_model(MODEL_FNAME)
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

compute_roc(train_visual_words, test_visual_words, train_labels, test_labels, classifier, RESULTS_DIR+'ROC.png')

accuracy = classifier.score(test_visual_words, test_labels)

print(f'Test accuracy: {accuracy}')

save_confusion_matrix(test_labels, classifier.predict(test_visual_words), RESULTS_DIR+'confusion_matrix.png')