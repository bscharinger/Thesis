from tensorflow import keras
from keras import Input, layers, Model
import tensorflow as tf

def get_model(img_size):

    inputs = Input(shape=img_size + (1,),dtype=tf.float16)
    print(inputs.shape)

    # [First half of the network: downsampling inputs] #

    # Entry block
    x = layers.Conv3D(32, 3, strides=2, padding="same", input_shape=(1, 128, 128, 128, 1))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256, 512]:
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling3D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv3D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.concatenate([x, residual], axis=-1)  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # [Second half of the network: upsampling inputs] #

    for filters in [256, 128, 64, 32]:
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.UpSampling3D(2)(x)

        # Project residual
        residual = layers.UpSampling3D(2)(previous_block_activation)
        residual = layers.Conv3D(filters, 1, padding="same")(residual)
        x = layers.concatenate([x, residual], axis=-1)  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    econv = layers.Conv3D(1, 3, padding='same')(x)
    outputs = layers.Activation('sigmoid', dtype='float32')(econv)
    # Define the model
    model = keras.Model(inputs, outputs)
    return model
