import tensorflow as tf
from keras import Input, layers, Model

def get_model(img_size):

    inputs = Input(shape=img_size + (1,),dtype=tf.float16)
    print(inputs.shape)

    # [First half of the network: downsampling inputs]

    # Entry block
    e1 = layers.Conv3D(16, 3, strides=1, padding="same", input_shape=(1, 64, 64, 64, 1))(inputs)
    e2 = layers.BatchNormalization()(e1)
    e3 = layers.Activation("relu")(e2)

    e4 = layers.Conv3D(16,3, strides=1, padding='same')(e3)
    e5 = layers.BatchNormalization()(e4)
    e6 = layers.Activation("relu")(e5)

    pool_e = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(e6)

    #Down Block 1
    db1conv1 = layers.Conv3D(32,3, strides=1, padding='same')(pool_e)
    db1bn1 = layers.BatchNormalization()(db1conv1)
    db1act1 = layers.Activation("relu")(db1bn1)

    db1conv2 = layers.Conv3D(32,3, strides=1, padding='same')(db1act1)
    db1bn2 = layers.BatchNormalization()(db1conv2)
    db1act2 = layers.Activation("relu")(db1bn2)

    pool_1 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(db1act2)

    #Down Block 2
    db2conv1 = layers.Conv3D(64,3, strides=1, padding='same')(pool_1)
    db2bn1 = layers.BatchNormalization()(db2conv1)
    db2act1 = layers.Activation("relu")(db2bn1)

    db2conv2 = layers.Conv3D(64,3, strides=1, padding='same')(db2act1)
    db2bn2 = layers.BatchNormalization()(db2conv2)
    db2act2 = layers.Activation("relu")(db2bn2)


    pool_2 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(db2act2)

    # Down Block 3
    db3conv1 = layers.Conv3D(128, 3, strides=1, padding='same')(pool_2)
    db3bn1 = layers.BatchNormalization()(db3conv1)
    db3act1 = layers.Activation("relu")(db3bn1)

    db3conv2 = layers.Conv3D(128, 3, strides=1, padding='same')(db3act1)
    db3bn2 = layers.BatchNormalization()(db3conv2)
    db3act2 = layers.Activation("relu")(db3bn2)

    pool_3 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(db3act2)

    #Bottom Block
    db4conv1 = layers.Conv3D(256,3, strides=1, padding='same')(pool_3)
    db4bn1 = layers.BatchNormalization()(db4conv1)
    db4act1 = layers.Activation("relu")(db4bn1)

    db4conv2 = layers.Conv3D(256,3, strides=1, padding='same')(db4act1)
    db4bn2 = layers.BatchNormalization()(db4conv2)
    db4act2 = layers.Activation("relu")(db4bn2)

    # Upsampling Block 3
    up3up = layers.UpSampling3D(size=2)(db4act2)

    up3conc = layers.concatenate([up3up, db3act2], axis=-1)

    up3conv1 = layers.Conv3D(128, 3, strides=1, padding="same")(up3conc)
    up3bn1 = layers.BatchNormalization()(up3conv1)
    up3act1 = layers.Activation("relu")(up3bn1)

    up3conv2 = layers.Conv3D(128, 3, strides=1, padding="same")(up3act1)
    up3bn2 = layers.BatchNormalization()(up3conv2)
    up3act2 = layers.Activation("relu")(up3bn2)

    #Upsampling Block 2
    up2up = layers.UpSampling3D(size=2)(up3act2)

    up2conc = layers.concatenate([up2up, db2act2], axis=-1)

    up2conv1 = layers.Conv3D(64,3,strides=1, padding="same")(up2conc)
    up2bn1 = layers.BatchNormalization()(up2conv1)
    up2act1 = layers.Activation("relu")(up2bn1)

    up2conv2 = layers.Conv3D(64,3,strides=1, padding="same")(up2act1)
    up2bn2 = layers.BatchNormalization()(up2conv2)
    up2act2 = layers.Activation("relu")(up2bn2)

    #Upsampling Block 1
    up1up = layers.UpSampling3D(size=2)(up2act2)

    up1conc = layers.concatenate([up1up, db1act2], axis=-1)

    up1conv1 = layers.Conv3D(32,3,strides=1, padding="same")(up1conc)
    up1bn1 = layers.BatchNormalization()(up1conv1)
    up1act1 = layers.Activation("relu")(up1bn1)

    up1conv2 = layers.Conv3D(32,3,strides=1, padding="same")(up1act1)
    up1bn2 = layers.BatchNormalization()(up1conv2)
    up1act2 = layers.Activation("relu")(up1bn2)

    #Upsampling Block 0
    up0up = layers.UpSampling3D(size=2)(up1act2)

    up0conc = layers.concatenate([up0up, e6], axis=-1)

    up0conv1 = layers.Conv3D(16,3,strides=1, padding="same")(up0conc)
    up0bn1 = layers.BatchNormalization()(up0conv1)
    up0act1 = layers.Activation("relu")(up0bn1)

    up0conv2 = layers.Conv3D(16,3,strides=1, padding="same")(up0act1)
    up0bn2 = layers.BatchNormalization()(up0conv2)
    up0act2 = layers.Activation("relu")(up0bn2)

    # Exit Layer
    econv = layers.Conv3D(1, 1, data_format="channels_last")(up0act2)
    outputs = layers.Activation('sigmoid', dtype='float32')(econv)

    model = Model(inputs, outputs)
    return model
