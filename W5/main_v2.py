from sklearn import metrics
from sklearn.utils import shuffle
import tensorflow as tf
from collections import Counter
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.layer_utils import count_params
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# from squeeze_net import make_model
# from sXception import make_model
# from baseline import make_model
from best_baseline import make_model
# from ours import make_model
# from ours_v2 import make_model

save = True

MODEL_NAME = 'predictionMap_test'
DATASET_DIR =  "MIT_split"
IMG_SIZE    = 128
BATCH_SIZE  = 16
epochs = 100
initial_lr = 0.01

def scheduler(epoch):
    if epoch < 3:
        return 1e-6 * (epoch+1)
    else:
        return 1e-3

def lr_exp_decay(epoch):
    k = 0.01
    return initial_lr * np.exp(-k*epoch)

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
# train_generator = train_datagen.flow_from_directory(
#         DATASET_DIR+'/train',  # this is the target directory
#         target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
#         batch_size=BATCH_SIZE,
#         classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
#         class_mode='categorical',
#         subset='training',
#         shuffle=True
#         )  # since we use binary_crossentropy loss, we need categorical labels

# validation_generator = train_datagen.flow_from_directory(
#         DATASET_DIR+'/train',
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=BATCH_SIZE,
#         classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
#         class_mode='categorical',
#         subset='validation')

# # this is a similar generator, for validation data
# test_generator = test_datagen.flow_from_directory(
#         DATASET_DIR+'/test',
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=BATCH_SIZE,
#         classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
#         class_mode='categorical')
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR+"/train",
    labels="inferred",
    label_mode="int",
    seed=1337,
    image_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR+"/test",
    labels="inferred",
    label_mode="int",
    seed=1337,
    image_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
)


model = make_model(input_shape=[IMG_SIZE, IMG_SIZE, 3])

# get number of parameters
trainable_count = count_params(model.trainable_weights)
non_trainable_count = count_params(model.non_trainable_weights)

print(model.summary())
if save:
    keras.utils.plot_model(model, show_shapes=True, to_file='results/' + MODEL_NAME + '.png')
    # Open the file
    with open('results/' + MODEL_NAME + '_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
        fh.close()

# counter = Counter(train_ds.classes)
# max_val = float(max(counter.values()))
# class_weight = {class_id : max_val/num_images for class_id, num_images in counter.items()}
# print('class weights: ', class_weight)

model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

callback = []
callback.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10))
callback.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

train_history=model.fit_generator(train_ds, 
                        epochs=epochs, 
                        validation_data=val_ds,
                        # class_weight = class_weight,
                        callbacks=callback
                        )



#Plot accuracy evolution
accuracy = train_history.history['accuracy']
val_accuracy = train_history.history['val_accuracy']

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

plt.title('Accuracy Evolution')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.plot(accuracy,label='Training')
plt.plot(val_accuracy,label='Validation')
plt.legend(loc='lower right')
if save:
    plt.savefig('results/' + MODEL_NAME + '_accuracy_evolution.png')
plt.close()

plt.title('Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.plot(loss,label='Training')
plt.plot(val_loss,label='Validation')
plt.legend(loc='upper left')
if save:
    plt.savefig('results/' + MODEL_NAME + '_loss_evolution.png')
plt.close()

#Evaluating model performance on test set
test_loss,test_accuracy=model.evaluate(val_ds)

print('test_loss',test_loss)
print('test_accuracy',test_accuracy)

performance_ratio = 100*test_accuracy/((trainable_count + non_trainable_count)/100000)
print('performance_ratio',performance_ratio)

if save:
    if test_accuracy > 0.85:
        model.save('results/' + MODEL_NAME + '.h5', overwrite=True)
    L = ['test_loss: ',str(test_loss),'\n','test_accuracy: ',str(test_accuracy),'\n','performance_ratio: ',str(performance_ratio)]
    with open('results/' + MODEL_NAME + '_summary.txt','a') as fh:
        fh.writelines(L)
        fh.close