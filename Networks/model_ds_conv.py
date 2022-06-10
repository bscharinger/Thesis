from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

def get_model(img_size):
    inputs = keras.Input(shape=img_size + (1,), dtype=tf.float16)
    print(inputs.shape)

    # [First half of the network: downsampling inputs]

    # Entry block
    e1 = layers.Conv3D(128, 7, strides=2, padding="same", input_shape=(1, 64, 64, 64, 1))(inputs)
    e2 = layers.BatchNormalization()(e1)
    e3 = layers.Activation("relu")(e2)
    pool1 = layers.MaxPool3D(pool_size=3, strides=2, padding='same')(e3)

    entry_block = e2  # Set aside residual

    # Dense Block 1
    db1conv1 = layers.Conv3D(128, 1, strides=1, padding="same")(pool1)
    db1conv1 = layers.BatchNormalization()(db1conv1)
    db1conv1 = layers.Activation("relu")(db1conv1)
    db1conc1 = layers.concatenate([pool1, db1conv1], axis=4)
    db1dsconv1 = layers.Conv3D(128, 3, strides=1, padding="same", groups=64)(db1conc1)
    db1dsconv2 = layers.Conv3D(32, 1, strides=1, padding="same")(db1dsconv1)
    db1ds = layers.BatchNormalization()(db1dsconv2)
    db1ds = layers.Activation("relu")(db1ds)
    db1 = layers.concatenate([pool1, db1conv1, db1ds], axis=4)
    dense_block_1 = db1

    # Transition 1
    tr1conv1 = layers.Conv3D(32, 1, strides=1, padding="same")(dense_block_1)
    tr1conv1 = layers.BatchNormalization()(tr1conv1)
    tr1conv1 = layers.Activation("relu")(tr1conv1)

    tr1conv2 = layers.Conv3D(32, 1, strides=2, padding="same")(tr1conv1)

    # Dense Block 2
    db2conv1 = layers.Conv3D(128, 1, strides=1, padding="same")(tr1conv2)
    db2conv1 = layers.BatchNormalization()(db2conv1)
    db2conv1 = layers.Activation("relu")(db2conv1)
    db2conc1 = layers.concatenate([tr1conv2, db2conv1], axis=4)
    db2dsconv1 = layers.Conv3D(160, 3, strides=1, padding="same", groups=160)(db2conc1)
    db2dsconv2 = layers.Conv3D(32, 1, strides=1, padding="same")(db2dsconv1)
    db2ds = layers.BatchNormalization()(db2dsconv2)
    db2ds = layers.Activation("relu")(db2ds)
    db2 = layers.concatenate([tr1conv2, db2conv1, db2ds], axis=4)
    dense_block_2 = db2

    # Transition 2
    tr2conv1 = layers.Conv3D(32, 1, strides=1, padding="same")(dense_block_2)
    tr2conv1 = layers.BatchNormalization()(tr2conv1)
    tr2conv1 = layers.Activation("relu")(tr2conv1)

    tr2conv2 = layers.Conv3D(32, 1, strides=2, padding="same")(tr2conv1)

    # Dense Block 3
    db3conv1 = layers.Conv3D(128, 1, strides=1, padding="same")(tr2conv2)
    db3conv1 = layers.BatchNormalization()(db3conv1)
    db3conv1 = layers.Activation("relu")(db3conv1)
    db3conc1 = layers.concatenate([tr2conv2, db3conv1], axis=4)
    db3dsconv1 = layers.Conv3D(160, 3, strides=1, padding="same", groups=160)(db3conc1)
    db3dsconv2 = layers.Conv3D(32, 1, strides=1, padding="same")(db3dsconv1)
    db3ds = layers.BatchNormalization()(db3dsconv2)
    db3ds = layers.Activation("relu")(db3ds)
    db3 = layers.concatenate([tr2conv2, db3conv1, db3ds], axis=4)
    dense_block_3 = db3

    # Transition 3
    tr3conv1 = layers.Conv3D(32, 1, strides=1, padding="same")(dense_block_3)
    tr3conv1 = layers.BatchNormalization()(tr3conv1)
    tr3conv1 = layers.Activation("relu")(tr3conv1)

    tr3conv2 = layers.Conv3D(32, 1, strides=2, padding="same")(tr3conv1)

    # Dense Block 4
    db4conv1 = layers.Conv3D(32, 1, strides=1, padding="same")(tr3conv2)
    db4conv1 = layers.BatchNormalization()(db4conv1)
    db4conv1 = layers.Activation("relu")(db4conv1)
    db4conc1 = layers.concatenate([tr3conv2, db4conv1], axis=4)
    db4dsconv1 = layers.Conv3D(64, 3, strides=1, padding="same", groups=64)(db4conc1)
    db4dsconv2 = layers.Conv3D(32, 1, strides=1, padding="same")(db4dsconv1)
    db4ds = layers.BatchNormalization()(db4dsconv2)
    db4ds = layers.Activation("relu")(db4ds)
    db4 = layers.concatenate([tr3conv2, db4conv1, db4ds], axis=4)
    dense_block_4 = db4

    # Upsampling Block 1
    up1up = layers.UpSampling3D(size=2)(dense_block_4)

    up1conc = layers.concatenate([db3, up1up], axis=4)
    up1conv = layers.Conv3D(64, 3, strides=1, padding="same")(up1conc)
    up1 = layers.BatchNormalization()(up1conv)
    up1 = layers.Activation("relu")(up1)

    # Upsampling Block 2
    up2up = layers.UpSampling3D(size=2)(up1)
    up2conc = layers.concatenate([db2, up2up], axis=4)
    up2conv = layers.Conv3D(32, 3, strides=1, padding="same")(up2conc)
    up2 = layers.BatchNormalization()(up2conv)
    up2 = layers.Activation("relu")(up2)

    # Upsampling Block 3
    up3up = layers.UpSampling3D(size=2)(up2)
    up3conc = layers.concatenate([db1, up3up], axis=4)
    up3conv = layers.Conv3D(16, 3, strides=1, padding="same")(up3conc)
    up3 = layers.BatchNormalization()(up3conv)
    up3 = layers.Activation("relu")(up3)

    # Upsampling Block 4
    up4up = layers.UpSampling3D(size=2)(up3)
    up4conc = layers.concatenate([entry_block, up4up], axis=4)
    up4conv = layers.Conv3D(8, 3, strides=1, padding="same")(up4conc)
    up4 = layers.BatchNormalization()(up4conv)
    up4 = layers.Activation("relu")(up4)

    # Upsampling Block 5
    up5up = layers.UpSampling3D(size=2)(up4)
    up5conv = layers.Conv3D(4, 3, strides=1, padding="same")(up5up)
    up5 = layers.BatchNormalization()(up5conv)
    up5 = layers.Activation("relu")(up5)

    # Exit Layer
    econv = layers.Conv3D(1, 1)(up5)
    outputs = layers.Activation('sigmoid', dtype='float32')(econv)

    model = keras.Model(inputs, outputs)
    return model
