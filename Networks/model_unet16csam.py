import keras.backend
import tensorflow as tf
from keras import Input, layers, Model

class SpatialAttention(layers.Layer):
    def __init__(self, n_channels, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.gamma = tf.Variable(0.0001, dtype=tf.float16, trainable=True).numpy()
        self.query = keras.Sequential([layers.Conv3D(self.n_channels//8,(1,3,1), padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.key = keras.Sequential([layers.Conv3D(self.n_channels//8,(3,1,1), padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
        self.judge = keras.Sequential([layers.Conv3D(self.n_channels//8,(1,1,3), padding='same'), layers.BatchNormalization(), layers.Activation('relu')])
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
        Q = layers.Reshape((-1,W*H*D))(Q)
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

        Q = input
        Q = layers.Reshape((C,-1))(Q)
        Q = tf.transpose(Q, perm=[0, 2, 1])

        K = input
        K = layers.Reshape((C,-1))(K)

        J = input
        J = layers.Reshape((C,-1))(J)
        J = tf.transpose(J, perm=[0, 2, 1])

        V = input
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
    e1 = layers.Conv3D(16, 3, strides=1, padding="same", input_shape=(1, 512, 512, 32, 1))(inputs)
    e2 = layers.BatchNormalization()(e1)
    e3 = layers.Activation("relu")(e2)

    e4 = layers.Conv3D(16,3, strides=1, padding='same')(e3)
    e5 = layers.BatchNormalization()(e4)
    e6 = layers.Activation("relu")(e5)

    pool_e = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(e6)

    #Down Block 1
    db1conv1 = layers.Conv3D(16,3, strides=1, padding='same')(pool_e)
    db1bn1 = layers.BatchNormalization()(db1conv1)
    db1act1 = layers.Activation("relu")(db1bn1)

    db1conv2 = layers.Conv3D(16,3, strides=1, padding='same')(db1act1)
    db1bn2 = layers.BatchNormalization()(db1conv2)
    db1act2 = layers.Activation("relu")(db1bn2)

    pool_1 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(db1act2)

    #Down Block 2
    db2conv1 = layers.Conv3D(16,3, strides=1, padding='same')(pool_1)
    db2bn1 = layers.BatchNormalization()(db2conv1)
    db2act1 = layers.Activation("relu")(db2bn1)

    db2conv2 = layers.Conv3D(16,3, strides=1, padding='same')(db2act1)
    db2bn2 = layers.BatchNormalization()(db2conv2)
    db2act2 = layers.Activation("relu")(db2bn2)

    pool_2 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')(db2act2)



    #Bottom Block = attn bblock
    bot = AffinityAttention(16)(pool_2)
    bot_bn = layers.BatchNormalization()(bot)
    bot_sum = layers.Add()([bot_bn, pool_2])




    # Upsampling Block 2
    up2up = layers.UpSampling3D(size=2)(bot_sum)

    up2conc = layers.concatenate([up2up, db2act2], axis=-1)

    up2conv1 = layers.Conv3D(16, 3, strides=1, padding="same")(up2conc)
    up2bn1 = layers.BatchNormalization()(up2conv1)
    up2act1 = layers.Activation("relu")(up2bn1)

    up2conv2 = layers.Conv3D(16, 3, strides=1, padding="same")(up2act1)
    up2bn2 = layers.BatchNormalization()(up2conv2)
    up2act2 = layers.Activation("relu")(up2bn2)

    #Upsampling Block 1
    up1up = layers.UpSampling3D(size=2)(up2act2)

    up1conc = layers.concatenate([up1up, db1act2], axis=-1)

    up1conv1 = layers.Conv3D(16,3,strides=1, padding="same")(up1conc)
    up1bn1 = layers.BatchNormalization()(up1conv1)
    up1act1 = layers.Activation("relu")(up1bn1)

    up1conv2 = layers.Conv3D(16,3,strides=1, padding="same")(up1act1)
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
