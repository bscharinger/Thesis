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



    #Down Block 1
    db1conv1 = layers.Conv3D(16, 3, strides=1, padding='same')(e3)
    db1bn1 = layers.BatchNormalization()(db1conv1)
    skip1 = layers.Activation("relu")(db1bn1)

    db1conv2 = layers.Conv3D(16, 3, strides=1, padding='same')(skip1)
    db1bn2 = layers.BatchNormalization()(db1conv2)
    db1act3 = layers.Activation("relu")(db1bn2)
    db1conv3 = layers.Conv3D(16, 3, strides=1, padding='same')(db1act3)
    db1bn3 = layers.BatchNormalization()(db1conv3)
    db1act2 = layers.Activation("relu")(db1bn3)

    db1add1 = layers.Add()([skip1, db1act2])

    db1conv4 = layers.Conv3D(16, 3, strides=1, padding='same')(db1add1)
    db1bn4 = layers.BatchNormalization()(db1conv4)
    db1act4 = layers.Activation("relu")(db1bn4)
    db1conv5 = layers.Conv3D(16, 3, strides=1, padding='same')(db1act4)
    db1bn5 = layers.BatchNormalization()(db1conv5)
    db1act5 = layers.Activation("relu")(db1bn5)

    db1add2 = layers.Add()([skip1, db1act5])

    # Down Block 2
    db2conv1 = layers.Conv3D(16, 3, strides=2, padding='same')(db1add2)
    db2bn1 = layers.BatchNormalization()(db2conv1)
    skip2 = layers.Activation("relu")(db2bn1)

    db2conv2 = layers.Conv3D(16, 3, strides=1, padding='same')(skip2)
    db2bn2 = layers.BatchNormalization()(db2conv2)
    db2act3 = layers.Activation("relu")(db2bn2)
    db2conv3 = layers.Conv3D(16, 3, strides=1, padding='same')(db2act3)
    db2bn3 = layers.BatchNormalization()(db2conv3)
    db2act2 = layers.Activation("relu")(db2bn3)

    db2add1 = layers.Add()([skip2, db2act2])

    db2conv4 = layers.Conv3D(16, 3, strides=1, padding='same')(db2add1)
    db2bn4 = layers.BatchNormalization()(db2conv4)
    db2act4 = layers.Activation("relu")(db2bn4)
    db2conv5 = layers.Conv3D(16, 3, strides=1, padding='same')(db2act4)
    db2bn5 = layers.BatchNormalization()(db2conv5)
    db2act5 = layers.Activation("relu")(db2bn5)

    db2add2 = layers.Add()([skip2, db2act5])

    # Down Block 3
    db3conv1 = layers.Conv3D(16, 3, strides=2, padding='same')(db2add2)
    db3bn1 = layers.BatchNormalization()(db3conv1)
    skip3 = layers.Activation("relu")(db3bn1)

    db3conv2 = layers.Conv3D(16, 3, strides=1, padding='same')(skip3)
    db3bn2 = layers.BatchNormalization()(db3conv2)
    db3act3 = layers.Activation("relu")(db3bn2)
    db3conv3 = layers.Conv3D(16, 3, strides=1, padding='same')(db3act3)
    db3bn3 = layers.BatchNormalization()(db3conv3)
    db3act2 = layers.Activation("relu")(db3bn3)

    db3add1 = layers.Add()([skip3, db3act2])

    db3conv4 = layers.Conv3D(16, 3, strides=1, padding='same')(db3add1)
    db3bn4 = layers.BatchNormalization()(db3conv4)
    db3act4 = layers.Activation("relu")(db3bn4)
    db3conv5 = layers.Conv3D(16, 3, strides=1, padding='same')(db3act4)
    db3bn5 = layers.BatchNormalization()(db3conv5)
    db3act5 = layers.Activation("relu")(db3bn5)

    db3add2 = layers.Add()([skip3, db3act5])

    # Down Block 4
    db4conv1 = layers.Conv3D(16, 3, strides=2, padding='same')(db3add2)
    db4bn1 = layers.BatchNormalization()(db4conv1)
    skip4 = layers.Activation("relu")(db4bn1)

    db4conv2 = layers.Conv3D(16, 3, strides=1, padding='same')(skip4)
    db4bn2 = layers.BatchNormalization()(db4conv2)
    db4act3 = layers.Activation("relu")(db4bn2)
    db4conv3 = layers.Conv3D(16, 3, strides=1, padding='same')(db4act3)
    db4bn3 = layers.BatchNormalization()(db4conv3)
    db4act2 = layers.Activation("relu")(db4bn3)

    db4add1 = layers.Add()([skip4, db4act2])

    db4conv4 = layers.Conv3D(16, 3, strides=1, padding='same')(db4add1)
    db4bn4 = layers.BatchNormalization()(db4conv4)
    db4act4 = layers.Activation("relu")(db4bn4)
    db4conv5 = layers.Conv3D(16, 3, strides=1, padding='same')(db4act4)
    db4bn5 = layers.BatchNormalization()(db4conv5)
    db4act5 = layers.Activation("relu")(db4bn5)

    db4add2 = layers.Add()([skip4, db4act5])

    #Up Block 1
    ub1tconv1 = layers.Conv3DTranspose(16,2, strides=2)(db4add2)

    ub1conc = layers.concatenate([ub1tconv1, db3add2],axis=-1)

    ub1conv1 = layers.Conv3D(16, 3, strides=1, padding='same')(ub1conc)
    ub1bn1 = layers.BatchNormalization()(ub1conv1)
    upskip1 = layers.Activation("relu")(ub1bn1)



    ub1conv2 = layers.Conv3D(16, 3, strides=1, padding='same')(upskip1)
    ub1bn2 = layers.BatchNormalization()(ub1conv2)
    ub1act2 = layers.Activation("relu")(ub1bn2)
    ub1conv3 = layers.Conv3D(16, 3, strides=1, padding='same')(ub1act2)
    ub1bn3 = layers.BatchNormalization()(ub1conv3)
    ub1act2 = layers.Activation("relu")(ub1bn3)

    up1add = layers.Add()([upskip1, ub1act2])

    # Up Block 2
    ub2tconv1 = layers.Conv3DTranspose(16, 2, strides=2)(up1add)

    ub2conc = layers.concatenate([ub2tconv1, db2add2], axis=-1)

    ub2conv1 = layers.Conv3D(16, 3, strides=1, padding='same')(ub2conc)
    ub2bn1 = layers.BatchNormalization()(ub2conv1)
    upskip2 = layers.Activation("relu")(ub2bn1)

    ub2conv2 = layers.Conv3D(16, 3, strides=1, padding='same')(upskip2)
    ub2bn2 = layers.BatchNormalization()(ub2conv2)
    ub2act2 = layers.Activation("relu")(ub2bn2)
    ub2conv3 = layers.Conv3D(16, 3, strides=1, padding='same')(ub2act2)
    ub2bn3 = layers.BatchNormalization()(ub2conv3)
    ub2act2 = layers.Activation("relu")(ub2bn3)

    up2add = layers.Add()([upskip2, ub2act2])

    # Up Block 3
    ub3tconv1 = layers.Conv3DTranspose(16, 2, strides=2)(up2add)

    ub3conc = layers.concatenate([ub3tconv1, db1add2], axis=-1)

    ub3conv1 = layers.Conv3D(16, 3, strides=1, padding='same')(ub3conc)
    ub3bn1 = layers.BatchNormalization()(ub3conv1)
    upskip3 = layers.Activation("relu")(ub3bn1)

    ub3conv2 = layers.Conv3D(16, 3, strides=1, padding='same')(upskip3)
    ub3bn2 = layers.BatchNormalization()(ub3conv2)
    ub3act2 = layers.Activation("relu")(ub3bn2)
    ub3conv3 = layers.Conv3D(16, 3, strides=1, padding='same')(ub3act2)
    ub3bn3 = layers.BatchNormalization()(ub3conv3)
    ub3act2 = layers.Activation("relu")(ub3bn3)

    up3add = layers.Add()([upskip3, ub3act2])

    #Out-Block 1
    outconv1 = layers.Conv3D(16, 3, strides=1, padding='same')(up3add)
    outskip = layers.Activation("relu")(outconv1)

    outconv2 = layers.Conv3D(16, 3, strides=1, padding='same')(outskip)
    outbn1 = layers.BatchNormalization()(outconv2)
    outact1 = layers.Activation("relu")(outbn1)
    outconv3 = layers.Conv3D(16, 3, strides=1, padding='same')(outact1)
    outbn2 = layers.BatchNormalization()(outconv3)
    outact2 = layers.Activation("relu")(outbn2)

    outadd1 = layers.Add()([outskip, outact2])

    outconv4 = layers.Conv3D(16, 3, strides=1, padding='same')(outadd1)
    outbn3 = layers.BatchNormalization()(outconv4)
    outact3 = layers.Activation("relu")(outbn3)
    outconv5 = layers.Conv3D(16, 3, strides=1, padding='same')(outact3)
    outbn4 = layers.BatchNormalization()(outconv5)
    outact4 = layers.Activation("relu")(outbn4)

    outadd2 = layers.Add()([outskip, outact4])

    # Exit Layer
    econv = layers.Conv3D(1, 3, data_format="channels_last", padding='same')(outadd2)
    outputs = layers.Activation('sigmoid', dtype='float32')(econv)

    model = Model(inputs, outputs)
    return model
