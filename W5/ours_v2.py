import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Total params: 1,157,832
# test_loss 0.5387701988220215
# test_accuracy 0.8686493039131165
# performance_ratio 7.502377753535198

# Set up data augmentation preprocessing layers
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2, 0.2),
        layers.RandomContrast((0.1, 0.9)),
    ]
)

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.BatchNormalization()(x)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    block_activation_1 = x

    # Mid block
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(128, 1, strides=2, padding="same")(block_activation_1)
    x = layers.add([x, residual])  # Add back residual
    block_activation_2 = x  # Set aside next residual

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(256, 1, strides=2, padding="same")(block_activation_2)
    x = layers.add([x, residual])  # Add back residual
    block_activation_3 = x  # Set aside next residual

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(512, 1, strides=2, padding="same")(block_activation_3)
    x = layers.add([x, residual])  # Add back residual

    x = layers.SeparableConv2D(728, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(8, activation="softmax")(x)
    return keras.Model(inputs, outputs)