import keras.backend
import tensorflow as tf
from keras import Input, layers, Model

class SpatialAttention(layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.gamma = tf.Variable(0.0001, dtype=tf.float16, trainable=True).numpy()
        self.query = keras.Sequential([layers.Conv3D(8,(1,3,1), padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.key = keras.Sequential([layers.Conv3D(8,(3,1,1), padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.judge = keras.Sequential([layers.Conv3D(8,(1,1,3), padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.value = keras.Sequential([layers.Conv3D(self.n_channels, 1, padding='same'), layers.BatchNormalization(), layers.Activation('relu')])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            'n_channels': self.n_channels,
            'gamma': self.gamma,
            'query': self.query,
            'key': self.key,
            'judge': self.judge,
            'value': self.value,
        })
        return config

    def call(self, input):

        B, W, D, H, C = input.get_shape().as_list()

        Q = self.query(input)
        Q = layers.Reshape((-1,W * H * D))(Q)
        Q = tf.transpose(Q, perm=[0,2,1])

        K = self.key(input)
        K = layers.Reshape((-1, W * H * D))(K)

        J = self.judge(input)
        J = layers.Reshape((-1, W * H * D))(J)
        J = tf.transpose(J, perm=[0, 2, 1])

        V = self.value(input)
        V = layers.Reshape((  W * H * D,-1))(V)

        m1 = tf.matmul(Q,K)
        m2 = tf.matmul(J,K)
        m3 = tf.matmul(m1,m2)

        affinity = layers.Activation(activation='sigmoid')(m3)

        weights = tf.matmul(affinity, V)
        weights = layers.Reshape((W,D,H,C))(weights)

        return tf.multiply(self.gamma, weights) + input


class ChannelAttention(layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.gamma = tf.Variable(0.0001, dtype=tf.float16, trainable=True).numpy()
        self.query = keras.Sequential([layers.Conv3D(self.n_channels, 1, padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.key = keras.Sequential([layers.Conv3D(self.n_channels, 1, padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.judge = keras.Sequential([layers.Conv3D(self.n_channels, 1, padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.value = keras.Sequential([layers.Conv3D(self.n_channels, 1, padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'n_channels': self.n_channels,
            'gamma': self.gamma,
            'query': self.query,
            'key': self.key,
            'judge': self.judge,
            'value': self.value,
        })
        return config

    def call(self, input):
        B, W, D, H, C = input.get_shape().as_list()

        Q = self.query(input)
        Q = layers.Reshape((C,-1))(Q)
        Q = tf.transpose(Q, perm=[0, 2, 1])

        K = self.key(input)
        K = layers.Reshape((C,-1))(K)

        J = self.judge(input)
        J = layers.Reshape((C,-1))(J)
        J = tf.transpose(J, perm=[0, 2, 1])

        V = self.value(input)
        V = layers.Reshape((C,-1))(V)

        m1 = tf.matmul(K,Q)
        m2 = tf.matmul(K,J)
        m3 = tf.matmul(m1, m2)

        affinity = layers.Activation(activation='sigmoid')(m3)

        weights = tf.matmul(affinity, V)
        weights = layers.Reshape((W, D, H, C))(weights)

        return tf.multiply(self.gamma, weights) + input

class AffinityAttention(layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(AffinityAttention, self).__init__(**kwargs)
        self.spatial = SpatialAttention(n_channels)
        self.channel = ChannelAttention(n_channels)
    def get_config(self):
        config = super(AffinityAttention, self).get_config()
        config.update({
            'spatial': self.spatial,
            'channel': self.channel,
        })
        return config
    def call(self, input):
        spat = self.spatial(input)
        chan = self.channel(input)
        return spat+chan+input

def get_model(img_size):

    inputs = Input(shape=img_size + (1,),dtype=tf.float16)
    print(inputs.shape)

    # [First half of the network: downsampling inputs]

    # Entry block
    residual = layers.Conv3D(16, 1, strides=1, padding="same", input_shape=(1, 256, 256, 128, 1))(inputs)

    e1 = layers.Conv3D(16, 3, strides=1, padding="same", input_shape=(1, 256, 256, 128, 1))(inputs)
    e2 = layers.BatchNormalization()(e1)
    e3 = layers.Activation("relu")(e2)

    e4 = layers.Conv3D(16,3, strides=1, padding='same')(e3)
    e5 = layers.BatchNormalization()(e4)
    e6 = layers.Activation("relu")(e5)

    e7 = layers.Add()([e6, residual])

    pool_e = layers.MaxPool3D()(e7)

    #Down Block 1
    residual = layers.Conv3D(16,1, strides=1, padding='same')(pool_e)

    db1conv1 = layers.Conv3D(16,3, strides=1, padding='same')(pool_e)
    db1bn1 = layers.BatchNormalization()(db1conv1)
    db1act1 = layers.Activation("relu")(db1bn1)

    db1conv2 = layers.Conv3D(16,3, strides=1, padding='same')(db1act1)
    db1bn2 = layers.BatchNormalization()(db1conv2)
    db1act2 = layers.Activation("relu")(db1bn2)

    db1add1 = layers.Add()([db1act2, residual])

    pool_1 = layers.MaxPool3D()(db1add1)

    #Down Block 2
    residual = layers.Conv3D(16,1, strides=1, padding='same')(pool_1)

    db2conv1 = layers.Conv3D(16,3, strides=1, padding='same')(pool_1)
    db2bn1 = layers.BatchNormalization()(db2conv1)
    db2act1 = layers.Activation("relu")(db2bn1)

    db2conv2 = layers.Conv3D(16,3, strides=1, padding='same')(db2act1)
    db2bn2 = layers.BatchNormalization()(db2conv2)
    db2act2 = layers.Activation("relu")(db2bn2)

    db2add1 = layers.Add()([db2act2, residual])

    pool_2 = layers.MaxPool3D()(db2add1)



    #Bottom Block = attn bblock
    bot = AffinityAttention(16)(pool_2)
    bot_bn = layers.BatchNormalization()(bot)
    bot_sum = layers.Add()([bot_bn, pool_2])




    # Upsampling Block 2
    up2up = layers.UpSampling3D(size=2)(bot_sum)
    #up2up = layers.Conv3DTranspose(16, 2, 2, padding='same')(bot_sum)

    up2conc = layers.concatenate([up2up, db2act2], axis=-1)

    residual = layers.Conv3D(16, 1, strides=1, padding="same")(up2conc)

    up2conv1 = layers.Conv3D(16, 3, strides=1, padding="same")(up2conc)
    up2bn1 = layers.BatchNormalization()(up2conv1)
    up2act1 = layers.Activation("relu")(up2bn1)

    up2conv2 = layers.Conv3D(16, 3, strides=1, padding="same")(up2act1)
    up2bn2 = layers.BatchNormalization()(up2conv2)
    up2act2 = layers.Activation("relu")(up2bn2)

    up2add1 = layers.Add()([up2act2, residual])

    #Upsampling Block 1
    up1up = layers.UpSampling3D(size=2)(up2add1)
    #up1up = layers.Conv3DTranspose(16,2,2,padding='same')(up2add1)

    up1conc = layers.concatenate([up1up, db1act2], axis=-1)

    residual = layers.Conv3D(16, 1, strides=1, padding="same")(up1conc)

    up1conv1 = layers.Conv3D(16,3,strides=1, padding="same")(up1conc)
    up1bn1 = layers.BatchNormalization()(up1conv1)
    up1act1 = layers.Activation("relu")(up1bn1)

    up1conv2 = layers.Conv3D(16,3,strides=1, padding="same")(up1act1)
    up1bn2 = layers.BatchNormalization()(up1conv2)
    up1act2 = layers.Activation("relu")(up1bn2)

    up1add1 = layers.Add()([up1act2, residual])

    #Upsampling Block 0
    up0up = layers.UpSampling3D(size=2)(up1add1)
    #up0up = layers.Conv3DTranspose(16, 2, 2, padding='same')(up1add1)

    up0conc = layers.concatenate([up0up, e6], axis=-1)

    residual = layers.Conv3D(16, 1, strides=1, padding="same")(up0conc)

    up0conv1 = layers.Conv3D(16,3,strides=1, padding="same")(up0conc)
    up0bn1 = layers.BatchNormalization()(up0conv1)
    up0act1 = layers.Activation("relu")(up0bn1)

    up0conv2 = layers.Conv3D(16,3,strides=1, padding="same")(up0act1)
    up0bn2 = layers.BatchNormalization()(up0conv2)
    up0act2 = layers.Activation("relu")(up0bn2)

    up0add1 = layers.Add()([up0act2, residual])

    # Exit Layer
    econv = layers.Conv3D(1, 1, data_format="channels_last")(up0add1)
    outputs = layers.Activation('sigmoid', dtype='float32')(econv)

    model = Model(inputs, outputs)
    return model
