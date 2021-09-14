# -*- coding: utf-8 -*-
"""AdaIN2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19GRTEMRfd-yhgzK3R-Oz9mC1daKc4xA-
"""
"""
!pip install tensorflow-addons

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/bshillingford/python-torchfile.git
# %cd python-torchfile/
!python setup.py install
# %cd ..

!wget -c https://s3.amazonaws.com/xunhuang-public/adain/vgg_normalised.t7
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


import os
import math
import time
from pathlib import Path
from PIL import Image


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torchfile


def load_and_process_img(path_to_img):
  """Load the image and preprocess according to trained VGG19 model standards.
  """
  new_min_dim = 512
  img = tf.io.read_file(path_to_img)
  # This creates RGB image
  try:
    img = tf.image.decode_jpeg(img, channels=3)
  except:
    return None

  # Scale minimum dimension to 512px
  height = tf.cast(tf.shape(img)[0], tf.float32)
  width = tf.cast(tf.shape(img)[1], tf.float32)
  min_dim = tf.minimum(height, width)
  scale = new_min_dim / min_dim
  img = tf.image.resize(img, (scale*height, scale*width))

  # This scales pixel values and reorders channels to BGR
  #img = tf.keras.applications.vgg19.preprocess_input(img)
  # img = tf.image.convert_image_dtype(img, tf.float32)
  # img = tf.cast(img, tf.float32)
  img /= 255.

  return img


def preprocess_img(img):
  """Preprocess image."""
  crop_size = 256
  img = tf.image.random_crop(img, (crop_size, crop_size, 3))

  return img


def process_path(path_to_img):
  img = load_and_process_img(path_to_img)
  img = preprocess_img(img)
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)


def load_img_Inference(path_to_img, max_dim=None, resize=True, frame = False):

    img = tf.io.read_file(path_to_img)     
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if resize:
        if frame:
            new_shape = tf.cast([218, 512], tf.int32)
            img = tf.image.resize(img, new_shape)
        else:  
            new_shape = tf.cast([256, 256], tf.int32)
            img = tf.image.resize(img, new_shape)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
    img = img[tf.newaxis, :]


    return img

BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE

def prepare_dataset(data):
  data = data.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # data = data.repeat()
  data = data.batch(BATCH_SIZE)
  data = data.prefetch(AUTOTUNE)

  return data

style_path ="/home/litsos/style_transfer/dataset/wikiart" #"/home/michlist/Desktop/Style_Transfer/Tensorflow2_Style_Transfer/dataset"
content_path = "/home/litsos/style_transfer/dataset/train2014"

NUM_STYLE_IMAGES = len(style_path + '/**/*.jpg')
NUM_CONTENT_IMAGES = len(content_path + '/*.jpg')
NUM_STYLE_BATCHES = math.ceil(NUM_STYLE_IMAGES / BATCH_SIZE)
NUM_CONTENT_BATCHES = math.ceil(NUM_CONTENT_IMAGES / BATCH_SIZE)

# Style images
style_dataset = tf.data.Dataset.list_files(style_path + '/**/*.jpg')
style_dataset = prepare_dataset(style_dataset)
# Content images
content_dataset = tf.data.Dataset.list_files(content_path + '/*.jpg')
content_dataset = prepare_dataset(content_dataset)

training_ds = tf.data.Dataset.zip((style_dataset, content_dataset))
style_tr, content_tr = next(iter(training_ds))

# Content layer where will pull our feature maps
content_layers = ['conv4_1'] 

# Style layer we are interested in
style_layers = ['conv1_1',
                'conv2_1',
                'conv3_1', 
                'conv4_1' 
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Import torch model into tensorflow
def get_encoder_from_torch(target_layer='relu4_1'):
  """Load a model from t7 and translate it to tensorflow."""
  t7 = torchfile.load('./vgg_normalised.t7', force_8bytes_long=True)

  inputs = tf.keras.Input((None, None, 3), name="vgg_input")

  x = inputs
    
  style_outputs = []
  content_outputs = []
  for idx,module in enumerate(t7.modules):
    name = module.name.decode() if module.name is not None else None
    
    if idx == 0:
      name = 'preprocess'  # VGG 1st layer preprocesses with a 1x1 conv to multiply by 255 and subtract BGR mean as bias

    if module._typename == b'nn.SpatialReflectionPadding':
      x = tf.keras.layers.Lambda(
          lambda t: tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]],
          mode='REFLECT'))(x)            
    elif module._typename == b'nn.SpatialConvolution':
      filters = module.nOutputPlane
      kernel_size = module.kH
      weight = module.weight.transpose([2,3,1,0])
      bias = module.bias
      x = layers.Conv2D(filters, kernel_size, padding='valid', activation='relu', name=name,
                    kernel_initializer=tf.constant_initializer(weight),
                    bias_initializer=tf.constant_initializer(bias),
                    trainable=False)(x)
      if name in style_layers:
        style_outputs.append(x)
      if name in content_layers:
        content_outputs.append(x)
    elif module._typename == b'nn.ReLU':
      pass # x = layers.Activation('relu', name=name)(x)
    elif module._typename == b'nn.SpatialMaxPooling':
      x = layers.MaxPooling2D(padding='same', name=name)(x)
    else:
      raise NotImplementedError(module._typename)

    if name == target_layer:
      # print("Reached target layer", target_layer)
      break
  
  # Get output layers corresponding to style and content layers 
  #style_outputs = [vgg.get_layer(name).output for name in style_layers]
  #content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs

  return models.Model(inputs=inputs, outputs=model_outputs)

def get_encoder():
  """ Creates encoder from VGG19 model.
  
  This function will load the VGG19 model and access the intermediate layers. 
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model. 
  
  Returns:
    returns a keras model that takes image inputs and outputs the style and 
      content intermediate layers. 
  """
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)

def get_decoder(encoder):
  """Creates a trainable decoder, that mirrors the encoder.

  Pooling layers are replaced with nearest up-sampling layers and reflection
  padding is used to avoid border artifacts.
  """
  decoder = tf.keras.Sequential()
  
  inputs = tf.keras.Input((None, None, encoder.layers[-1].filters))
  # Mirror the encoder
  x = inputs
  for i in reversed(range(4, len(encoder.layers))):
    layer = encoder.layers[i]
    if isinstance(layer, layers.MaxPooling2D):
      x = layers.UpSampling2D()(x)
    elif isinstance(layer, layers.Conv2D):
      x = layers.Conv2D(
          layer.get_weights()[0].shape[2], 
          layer.kernel_size, 
          activation=tf.keras.activations.relu)(
              tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]],
              mode='REFLECT'))

  # Finally reduce number of channels to three
  x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]],
             mode='REFLECT')
  x = tf.keras.layers.Conv2D(3, 
                             3)(x) 
                             # activation=tf.keras.activations.relu)(x)
  outputs = x
    
  return models.Model(inputs, outputs)

def adaptive_instance_normalization(x, y):
  """Aligning the mean and variance of y onto x."""
  eps = 1e-4
  x_mean, x_var = tf.nn.moments(x, [1,2], keepdims=True)
  x_std = tf.math.sqrt(x_var)
  y_mean, y_var = tf.nn.moments(y, [1,2], keepdims=True)
  y_std = tf.math.sqrt(y_var)
  # result = y_std * (x - x_mean) / (x_std + eps) + y_mean 
  result = tf.nn.batch_normalization(x, x_mean, x_std, y_mean, y_std, eps)
  return result

encoder = get_encoder_from_torch()
decoder = get_decoder(encoder)

encoder.summary()

def get_content_loss(adain_output, target_encoded):
  return tf.reduce_mean(tf.square(adain_output - target_encoded))

def get_style_loss(base_style_encoded, target_encoded):
  eps = 1e-5
  
  base_style_mean, base_style_var = tf.nn.moments(base_style_encoded, 
                                                  axes=[1,2])
  # Add epsilon for numerical stability for gradients close to zero
  base_style_std = tf.math.sqrt(base_style_var + eps)

  target_mean, target_var = tf.nn.moments(target_encoded,
                                          axes=[1,2])
  # Add epsilon for numerical stability for gradients close to zero
  target_std = tf.math.sqrt(target_var + eps)

  mean_diff = tf.reduce_sum(tf.square(base_style_mean - target_mean)) / BATCH_SIZE
  std_diff = tf.reduce_sum(tf.square(base_style_std - target_std)) / BATCH_SIZE
  return mean_diff + std_diff

STYLE_LOSS_WEIGHT = 1

def get_loss(adain_output, base_style_encoded, target_encoded):
  # Content loss
  content_loss = get_content_loss(adain_output, target_encoded[-1])
  
  # Style loss
  style_loss = 0
  for i in range(num_style_layers):
    style_loss += get_style_loss(base_style_encoded[i], target_encoded[i])

  return content_loss + STYLE_LOSS_WEIGHT * style_loss

def decode_img(img, reverse_channels=False):
  """Decodes preprocessed images."""

  # perform the inverse of the preprocessiing step
  img *= 255.
  if reverse_channels:
    img = img[..., ::-1]

  img = tf.cast(img, dtype=tf.uint8)
  return img

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function
def train_step(content_img, style_img):
  with tf.GradientTape() as tape:
    encoded_content_img = encoder(content_img)
    encoded_style_img = encoder(style_img)
    tape.watch(encoded_content_img + encoded_style_img)

    adain_output = adaptive_instance_normalization(encoded_content_img[-1],
                                        encoded_style_img[-1])

    target_img = decoder(adain_output)

    loss = get_loss(adain_output, encoded_style_img, encoder(target_img))

    

  gradients = tape.gradient(loss, decoder.trainable_variables)
  optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

  train_loss(loss)

EPOCHS = 2
PROGBAR = tf.keras.utils.Progbar(len(training_ds))

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()

  step = 0
  start_time = time.perf_counter()
  for (style_tr, content_tr) in training_ds.as_numpy_iterator():
      start_time = time.perf_counter()
      train_step(content_tr, style_tr)
      print(f"Train step: {time.perf_counter() - start_time}")
      # start_time = time.perf_counter()
      step += 1
      PROGBAR.update(step)

  template = 'Epoch {}, Loss: {}'
  print(template.format(epoch+1,
                        train_loss.result()))
"""
style = process_path("/content/drive/MyDrive/content_images/Wikiart/Action_painting/franz-kline_accent-grave-1955.jpg")
content =process_path("/content/drive/MyDrive/content_images/mini_batch/000000000802.jpg")
style = style[tf.newaxis, :]
content = content[tf.newaxis, :]

encoded_content_img = encoder(content)
encoded_style_img = encoder(style)
adain_output = adaptive_instance_normalization(encoded_content_img[-1],
                                    encoded_style_img[-1])
target_img = decoder(adain_output)

"""