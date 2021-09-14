import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image

IMAGE_SIZE = [256, 256]

def load_img(path_to_img, max_dim=None, resize=True):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if resize:
        min_dim = 512
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        small_dim = tf.reduce_min(shape)
        scale = min_dim / small_dim
        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = tf.image.random_crop(img, [256,256,3])
        img = 255*img

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        
    img = img[tf.newaxis, :]

    return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def get_mini_vgg():
    vgg19 = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet'
    )
    vgg19.trainable = False
    mini_vgg19 = tf.keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output, name="mini_vgg19")
    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
    mini_vgg19_out = mini_vgg19(preprocessed)
    return tf.keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")

def get_loss_net():
    vgg19 = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet'
    )
    vgg19.trainable = False
    layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1"
    ]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = tf.keras.Model(vgg19.input, outputs)
    
    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
    mini_vgg19_out = mini_vgg19(preprocessed)
    return tf.keras.Model(inputs, mini_vgg19_out, name="loss_net")

"""#AdaIN Layer"""

# Reference: https://github.com/ftokarev/tf-adain
def ada_in(style, content, epsilon=1e-5):
    axes = [1,2]

    c_mean, c_var = tf.nn.moments(content, axes=axes, keepdims=True)
    s_mean, s_var = tf.nn.moments(style, axes=axes, keepdims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)
    
    t = s_std * (content - c_mean) / c_std + s_mean
    return t

"""#Generator Block"""

def upsample_nearest(inputs, scale):
    shape = tf.shape(inputs)
    n, h, w, c = shape[0], shape[1], shape[2], shape[3]
    return tf.image.resize(inputs, tf.stack([h*scale, w*scale]), method="nearest")

def get_decoder_block():
  inputs = tf.keras.layers.Input(shape=(None, None, 512), batch_size = None)
  x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same")(inputs)
  x = tf.keras.layers.ReLU()(x)
  x = upsample_nearest(x,2)

  x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same")(x)
  x = tf.keras.layers.ReLU()(x)
  
  x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
  x = tf.keras.layers.ReLU()(x)

  x = upsample_nearest(x,2)

  x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same")(x)
  x = tf.keras.layers.ReLU()(x)

  x = upsample_nearest(x,2)

  x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")(x)
  x = tf.keras.layers.ReLU()(x)

  x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), padding="same")(x)
  return tf.keras.Model(inputs=inputs, outputs=x, name="Decoder")

"""#Train"""

def get_mean_std(tensor, epsilon=1e-5):
    tensor_mean, tensor_var = tf.nn.moments(tensor, axes=[1,2], keepdims=True)
    tensor_std = tf.sqrt(tensor_var + epsilon)
    return tensor_mean, tensor_std

def compute_style_loss(style_image, generated_image):
    style_features_in_style_image = style_image
    style_features_in_gen_image = generated_image

    style_layer_loss = []
    for style_feat, gen_feat in zip(style_features_in_style_image ,style_features_in_gen_image):
        meanS, varS = tf.nn.moments(style_feat, [1,2])
        meanG, varG = tf.nn.moments(gen_feat, [1,2])

        sigmaS = tf.sqrt(varS + 1e-5)
        sigmaG = tf.sqrt(varG + 1e-5)

        l2_mean = tf.reduce_sum(tf.square(meanG - meanS))
        l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

        style_layer_loss.append(l2_mean + l2_sigma)

    return tf.reduce_sum(style_layer_loss)

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
  for i in range(4):
    style_loss += get_style_loss(base_style_encoded[i], target_encoded[i])

  return content_loss + STYLE_LOSS_WEIGHT * style_loss

def build_content_loss(current, target, weight):
    loss = tf.reduce_mean(tf.math.squared_difference(current, target))
    loss *= weight
    return loss

def build_style_losses(current_layers, target_layers, weight, epsilon=1e-6):
    losses = {}
    for layer in range(4):
        current, target = current_layers[layer], target_layers[layer]

        current_mean, current_var = tf.nn.moments(current, axes=[2,3], keepdims=True)
        current_std = tf.sqrt(current_var + epsilon)

        target_mean, target_var = tf.nn.moments(target, axes=[2,3], keepdims=True)
        target_std = tf.sqrt(target_var + epsilon)

        mean_loss = tf.reduce_sum(tf.math.squared_difference(current_mean, target_mean))
        std_loss = tf.reduce_sum(tf.math.squared_difference(current_std, target_std))

        # normalize w.r.t batch size
        n = tf.cast(tf.shape(current)[0], dtype=tf.float32)
        mean_loss /= n
        std_loss /= n

        losses[layer] = (mean_loss + std_loss) * weight
    return losses

class CustomModel(tf.keras.Model):
    def __init__(self, mini_vgg, loss_net, generator):
        super().__init__()
        self.mini_vgg = mini_vgg
        self.loss_net = loss_net
        self.generator = generator


    def compile(self, opt, loss_fn):
        super().compile()
        self.opt = opt
        self.loss_fn = loss_fn
        
    
    def train_step(self, data):
        style, content = data
        #print(style)
    
        with tf.GradientTape() as tape:
            # encode the content and style
            style_enc = self.mini_vgg(style, training=False)
            content_enc = self.mini_vgg(content, training= False)
            
            # build the adain output
            t = ada_in(style_enc, content_enc)
            
            # generate the stylised image
            output_image = self.generator(t, training=True)
            #output_image = tf.clip_by_value(output_image, 0.0, 255.0)
            output_vgg_loss = self.loss_net(output_image, training=True)
            style_vgg_loss = self.loss_net(style, training=True)
            #print(style_vgg_loss)
            #print(output_vgg_loss)
            
            #loss_content = build_content_loss(output_vgg_loss[-1], t, 1)
            #loss_style = build_style_losses(output_vgg_loss, style_vgg_loss, 10)
            total_loss = get_loss(t, style_vgg_loss, output_vgg_loss )
            #print(loss_style)
            
            # content loss
            #loss_content = self.loss_fn(t,output_vgg_loss[-1])
            #loss_content = tf.math.reduce_mean(loss_content, axis=[-1,-2])
            
            # style loss
            """
            loss_style = 0
            for inp, out in zip(style_vgg_loss, output_vgg_loss):
                mean_inp, std_inp = get_mean_std(inp)
                mean_out, std_out = get_mean_std(out)
                
                loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(std_inp, std_out)
            
            loss_style = tf.math.reduce_mean(loss_style, axis=[-1,-2])
            total_loss = loss_content + 4*loss_style
        
            """
            #total_loss = loss_content + tf.reduce_sum(list(loss_style.values()))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.opt.apply_gradients(zip(gradients, trainable_vars))
        
        return {
            "total_loss": total_loss
            #"content_loss": loss_content,
            #"style_loss": 10 * tf.reduce_sum(list(loss_style.values())),
        }

EPOCHS = 2
style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1"
    ]
  

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.mean_squared_error

encoder = get_mini_vgg()
loss_net = get_loss_net()
decoder = get_decoder_block()

model = CustomModel(
    mini_vgg = encoder,
    loss_net = loss_net,
    generator = decoder,
)
model.compile(
    opt = opt,
    loss_fn = loss_fn
)


model.load_weights("/home/michlist/Desktop/Style_Transfer/Tensorflow2_Style_Transfer/Adain/weights/model").expect_partial()
style = load_img("/home/michlist/Desktop/Style_Transfer/Tensorflow2_Style_Transfer/style_images/composition.jpg")/255
content = load_img("/home/michlist/Desktop/Style_Transfer/Tensorflow2_Style_Transfer/dataset/mini_batch/000000365766.jpg")/255

style_enc = model.mini_vgg(style, training=False)
content_enc = model.mini_vgg(content, training= False)
t = ada_in(style_enc, content_enc)
output_image = model.generator.predict(t)

print(output_image)

plt.imshow(tf.squeeze(content))
plt.show()