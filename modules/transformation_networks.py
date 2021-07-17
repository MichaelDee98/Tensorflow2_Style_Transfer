import tensorflow as tf
import tensorflow_addons as tfa


class instance_norm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3):
        super(instance_norm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, self.epsilon)))
        
        return self.gamma * x + self.beta

class ConvLayer(tf.keras.layers.Layer):
  def __init__(self, filters, 
               kernel=(3,3), padding='same', 
               strides=(1,1), activate=True, name="", 
               weight_initializer="glorot_uniform",
               inputs_shape = (None, None, None ,None)
               ):
    super(ConvLayer, self).__init__()
    self.activate = activate
    self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel, 
                       padding=padding, strides=strides, 
                       name=name, trainable=True,
                       use_bias=False, 
                       kernel_initializer=weight_initializer,
                       inputs_shape = input_shape)
    self.inst_norm = tfa.layers.InstanceNormalization(axis=3, 
                                          center=True, 
                                          scale=True, 
                                          beta_initializer="zeros", 
                                          gamma_initializer="ones",
                                          trainable=True)
    if self.activate:
      self.relu_layer = tf.keras.layers.Activation('relu', trainable=False)

  def call(self, x):
    x = self.conv(x)
    x = self.inst_norm(x)
    if self.activate:
      x = self.relu_layer(x)
    return x


class ResBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel=(3,3), padding='same', weight_initializer="glorot_uniform", prefix=""):
    super(ResBlock, self).__init__()
    self.prefix_name = prefix + "_"
    self.conv1 =ConvLayer(filters=filters, 
                           kernel=kernel, 
                           padding=padding, 
                           weight_initializer=weight_initializer,
                           name=self.prefix_name + "conv_1")
    self.conv2 =ConvLayer(filters=filters, 
                           kernel=kernel, 
                           padding=padding, 
                           activate=False, 
                           weight_initializer=weight_initializer,
                           name=self.prefix_name + "conv_2")
    self.add = tf.keras.layers.Add(name=self.prefix_name + "add")

  def call(self, x):
    tmp = self.conv1(x)
    c = self.conv2(tmp)
    return self.add([x, c])


class ConvTLayer(tf.keras.layers.Layer):
  def __init__(self, filters, kernel=(3,3), padding='same', strides=(1,1), activate=True, name="",
               weight_initializer="glorot_uniform" 
               ):
    super(ConvTLayer, self).__init__()
    self.activate = activate
    self.conv_t = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel, padding=padding, 
                                  strides=strides, name=name, 
                                  use_bias=False,
                                  kernel_initializer=weight_initializer)
    self.inst_norm =  tfa.layers.InstanceNormalization(axis=3, 
                                          center=True, 
                                          scale=True, 
                                          beta_initializer="zeros", 
                                          gamma_initializer="ones",
                                          trainable=True)
    if self.activate:
      self.relu_layer = tf.keras.layers.Activation('relu')

  def call(self, x):
    x = self.conv_t(x)
    x = self.inst_norm(x)
    if self.activate:
      x = self.relu_layer(x)
    return x

class TransformNet(tf.keras.models.Model):
  def __init__(self):
    super(TransformNet, self).__init__()
    self.conv1 = ConvLayer(32, (9,9), strides=(1,1), padding='same', name="conv_1", input_shape=(None,None,None,3))
    self.conv2 = ConvLayer(64, (3,3), strides=(2,2), padding='same', name="conv_2")
    self.conv3 = ConvLayer(128, (3,3), strides=(2,2), padding='same', name="conv_3")
    self.res1 = ResBlock(128, prefix="res_1")
    self.res2 = ResBlock(128, prefix="res_2")
    self.res3 = ResBlock(128, prefix="res_3")
    self.res4 = ResBlock(128, prefix="res_4")
    self.res5 = ResBlock(128, prefix="res_5")
    self.convt1 = ConvTLayer(64, (3,3), strides=(2,2), padding='same', name="conv_t_1")
    self.convt2 = ConvTLayer(32, (3,3), strides=(2,2), padding='same', name="conv_t_2")
    self.conv4 = ConvLayer(3, (9,9), strides=(1,1), padding='same', activate=False, name="conv_4")



  def call(self, inputs, crop = False):

    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.res5(x)
    x = self.convt1(x)
    x = self.convt2(x)
    x = self.conv4(x)
    if crop:
      x = x[:,:218,:,:]
    x = tf.nn.tanh(x)
    #x = (x + 1) / 2
    x=(x + 1) * (255. / 2)
    return x

class TransformNetAgrim(tf.keras.models.Model):
  def __init__(self):
    super(TransformNetAgrim, self).__init__()
    self.conv1 = ConvLayer(32, (9,9), strides=(1,1), padding='same', name="conv_1",input_shape=(None,None,None,6))
    self.conv2 = ConvLayer(64, (3,3), strides=(2,2), padding='same', name="conv_2")
    self.conv3 = ConvLayer(128, (3,3), strides=(2,2), padding='same', name="conv_3")
    self.res1 = ResBlock(128, prefix="res_1")
    self.res2 = ResBlock(128, prefix="res_2")
    self.res3 = ResBlock(128, prefix="res_3")
    self.res4 = ResBlock(128, prefix="res_4")
    self.res5 = ResBlock(128, prefix="res_5")
    self.convt1 = ConvTLayer(64, (3,3), strides=(2,2), padding='same', name="conv_t_1")
    self.convt2 = ConvTLayer(32, (3,3), strides=(2,2), padding='same', name="conv_t_2")
    self.conv4 = ConvLayer(3, (9,9), strides=(1,1), padding='same', activate=False, name="conv_4")



  def call(self, inputs1, inputs2, crop = False):
    x = tf.keras.layers.concatenate([inputs1, inputs2]) 
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.res5(x)
    x = self.convt1(x)
    x = self.convt2(x)
    x = self.conv4(x)
    if crop:
      x = x[:,:218,:,:]
    x = tf.nn.tanh(x)
    #x = (x + 1) / 2
    x=(x + 1) * (255. / 2)
    return x