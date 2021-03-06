import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(64,7,activation='relu',padding='same')(inputs)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128,3,activation='relu', padding='same')(x)
    # x = layers.Conv2D(128,3,activation='relu', padding='same')(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256,3,activation='relu', padding='same')(x)
    # x = layers.Conv2D(256,3,activation='relu', padding='same')(x)

    x = layers.AveragePooling2D(2)(x)
    x = layers.Flatten()(x)
    # x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(8, activation="softmax")(x)
    return keras.Model(inputs, outputs)