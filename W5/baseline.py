import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 34,679,496
# performance_ratio: 0.21

def make_model_layers(input_shape, netLayers):
    inputs = keras.Input(shape=input_shape)
    for layer in netLayers:
        if layer[0] == 'Dense':
            x = layers.Dense(layer[1], activation=layer[2])(x)
        elif layer[0] == 'Dropout':
            x = layers.Dropout(layer[1])(x)
        elif layer[0] == 'BatchNorm':
            x = layers.BatchNormalization()(x)
        elif layer[0] == 'Conv2D':
            x = layers.Conv2D(layer[1],layer[2],activation='relu',padding='same')(x)
        elif layer[0] == 'MaxPool':
            x = layers.MaxPooling2D(layer[1])(x)
        elif layer[0] == 'AvPool':
            x = layers.AveragePooling2D(layer[1])(x)

    outputs = layers.Dense(8, activation="softmax")(x)
    return keras.Model(inputs, outputs)
     


def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(64,7,activation='relu',padding='same')(inputs)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # x = layers.MaxPooling2D(2)(x)
    # x = layers.Activation("relu")(x)
    # x = layers.SeparableConv2D(728, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    # x = layers.SeparableConv2D(728, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)

    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dropout(0.2)(x)

    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(0.4)(x)

    # x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(8, activation="softmax")(x)
    return keras.Model(inputs, outputs)