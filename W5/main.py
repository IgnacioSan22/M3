import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.layer_utils import count_params
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# from squeeze_net import make_model
# from sXception import make_model
from baseline import make_model


MODEL_NAME  = 'baseline_dropout_2Convlayer_11kern'
DATASET_DIR =  "MIT_split"
IMG_SIZE    = 64
BATCH_SIZE  = 16

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical'
        )  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')

model = make_model(input_shape=[IMG_SIZE, IMG_SIZE, 3])
print(model.summary())
keras.utils.plot_model(model, show_shapes=True, to_file='results/' + MODEL_NAME + '.png')

# get number of parameters
trainable_count = count_params(model.trainable_weights)
non_trainable_count = count_params(model.non_trainable_weights)

model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

train_history=model.fit(train_generator, epochs=40, validation_data=validation_generator)

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
plt.savefig('results/' + MODEL_NAME + '_accuracy_evolution.png')
plt.close()

plt.title('Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.plot(loss,label='Training')
plt.plot(val_loss,label='Validation')
plt.legend(loc='upper left')
plt.savefig('results/' + MODEL_NAME + '_loss_evolution.png')
plt.close()

#Evaluating model performance on test set
test_loss,test_accuracy=model.evaluate(validation_generator)

print('test_loss',test_loss)
print('test_accuracy',test_accuracy)

performance_ratio = 100*test_accuracy/((trainable_count + non_trainable_count)/100000)
print('performance_ratio',performance_ratio)

# model.save_weights('results/' + MODEL_NAME + '.h5', overwrite=True)