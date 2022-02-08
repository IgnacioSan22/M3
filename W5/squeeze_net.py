from tensorflow import keras
from tensorflow.keras import layers

def make_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    conv1 = layers.Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',)(inputs)
    maxpool1 = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)
    fire2_squeeze = layers.Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = layers.Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = layers.Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = layers.Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze')(merge2)
    fire3_expand1 = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = layers.Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2')(fire3_squeeze)
    merge3 = layers.Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = layers.Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze')(merge3)
    fire4_expand1 = layers.Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = layers.Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = layers.Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

    fire5_squeeze = layers.Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = layers.Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = layers.Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = layers.Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = layers.Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = layers.Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = layers.Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2')(fire6_squeeze)
    merge6 = layers.Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = layers.Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze')(merge6)
    fire7_expand1 = layers.Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1')(fire7_squeeze)
    fire7_expand2 = layers.Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2')(fire7_squeeze)
    merge7 = layers.Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze')(merge7)
    fire8_expand1 = layers.Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1')(fire8_squeeze)
    fire8_expand2 = layers.Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2')(fire8_squeeze)
    merge8 = layers.Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    maxpool8 = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)
    fire9_squeeze = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze')(maxpool8)
    fire9_expand1 = layers.Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1')(fire9_squeeze)
    fire9_expand2 = layers.Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2')(fire9_squeeze)
    merge9 = layers.Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = layers.Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = layers.Convolution2D(
        8, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10')(fire9_dropout)

    global_avgpool10 = layers.GlobalAveragePooling2D()(conv10)
    softmax = layers.Activation("softmax", name='softmax')(global_avgpool10)

    return keras.Model(inputs=inputs, outputs=softmax)