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
     
data_augmentation = keras.Sequential(
    [
        # layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        # layers.RandomZoom(0.2, 0.2),
        # layers.RandomContrast((0.1, 0.9)),
    ]
)

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # Entry block
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,7,activation='relu',padding='same')(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128,3,activation='relu', padding='same')(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256,3,activation='relu', padding='same')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(8, activation="softmax")(x)
    return keras.Model(inputs, outputs)