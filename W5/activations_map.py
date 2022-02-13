import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate
import keras
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import keract

# model = MobileNetV2()
model = keras.models.load_model('results/best_model_separable_300epoch_noAug.h5')

image = Image.open('activations/land8/land28.jpg')
image = image.resize((128, 128))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# image = image / 255.

yhat = model.predict(image)
print(f'Predictions: {yhat}')
# label = decode_predictions(yhat)
# label = label[0][0]
# print('{} ({})'.format(label[1], label[2] * 100))

model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

activations = keract.get_activations(model, image)
first = activations.get('out_relu')
keract.display_activations(activations, save=True, directory='activations/land8')