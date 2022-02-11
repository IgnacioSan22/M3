import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 380,000
#test acc: 91.3
# performance_ratio: 23.97

data_augmentation = keras.Sequential(
    [
        # layers.RandomFlip("horizontal"),
        # layers.RandomRotation(0.1),
        layers.RandomZoom(0.2, 0.2),
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
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(128,3,activation='relu', padding='same')(x)
    # x = layers.Dropout(0.2)(x)

    x = layers.MaxPooling2D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(256,3,activation='relu', padding='same')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(8, activation="softmax")(x)
    return keras.Model(inputs, outputs)