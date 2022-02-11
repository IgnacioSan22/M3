import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle as Pickle
import numpy as np
from PIL import Image
from sklearn import metrics



MODEL_NAME = 'basemodel_layerNorm_AdamWarmlr'
DATASET_DIR =  "MIT_split"
IMG_SIZE    = 128


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR+"/test",
    labels="inferred",
    label_mode="int",
    shuffle=False,
    seed=1337,
    image_size=(IMG_SIZE,IMG_SIZE),
    batch_size=1,
)

# Show class names of the dataset
class_names = val_ds.class_names
print(class_names)
n_classes = len(class_names)
print("Number of classes: ", n_classes)

#Load trained model
model = keras.models.load_model('results/' + MODEL_NAME + '.h5')

y_true = []
y_pred = []
#Predict class for all test imaeges
for image,label in val_ds.take(-1):
  y_true.append(label.numpy()[0])
  y_p = model.predict(image)
  y_pred.append(np.argmax(y_p))

#Check class balance of learned model
metrics.balanced_accuracy_score(y_true,y_pred)

def visualize_wrong_predictions(image_filenames, labels, predictions, samples_per_class=5):
    print(f'Number of samples: {len(predictions)}')
    # print(f'Number of wrongly classified samples: {sum(predictions!=labels)}')
    
    # get unique classses
    classes = np.unique(np.array(labels))
    num_classes = len(classes)
    #set size for plot
    plt.figure(figsize=(24,16))
    
    def get_index_fp(idxs):
        items = []
        for idx in idxs:
            if predictions[idx] != labels[idx]:
                items.append(idx)
        return items
    
    for y, cls in enumerate(classes):
        idxs_all = np.flatnonzero(np.array(labels) == cls)
        idxs_fp = get_index_fp(idxs_all)
        idxs = np.random.choice(idxs_fp, samples_per_class, replace=True)
        class_accuracy = 100*(1 - (len(idxs_fp)/len(idxs_all)))
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(Image.open(image_filenames[idx]))
            plt.axis('off')
            labels_idx = int(labels[idx])
            predictions_idx = int(predictions[idx])
            if i == 0:
                plt.title(f'{class_names[y]} \n ClassAccuracy: {class_accuracy:.2f}  \n GT: {class_names[labels_idx]},\n Pred: {class_names[predictions_idx]}')
            else:
                plt.title(f'GT: {class_names[labels_idx]},\n Pred: {class_names[predictions_idx]}')
    plt.show()

#Load images to plot them
test_images_filenames = Pickle.load(open(DATASET_DIR + '/test_images_filenames.dat','rb'))
test_images_filenames  = [n[16:] for n in test_images_filenames]

visualize_wrong_predictions(test_images_filenames, y_true, y_pred,5)