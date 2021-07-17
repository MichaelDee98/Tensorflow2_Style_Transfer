import tensorflow as tf


# Set up the loss Network
def vgg_layers(layer_names):
    """ Creates the VGG19 model and returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, input, training = False):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(input)
        preprocessed_input = tf.cast(preprocessed_input, tf.float32)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                            outputs[self.num_style_layers:])
        if not training:
            style_outputs = [gram_matrix(style_output)
                            for style_output in style_outputs]


        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                        for style_name, value
                        in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def gram_matrix(input_tensor, shape=None):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = input_tensor.get_shape()
    num_locations = input_shape[1] * input_shape[2] * input_shape[3]
    num_locations = tf.cast(num_locations, tf.float32)
    return result / num_locations

def get_loss(transformed_content, transformed_style, content_targets, style_targets, transformed_img, style_weight, content_weight, tv_weight):

    content_loss = get_content_loss(transformed_content, content_targets)
    style_loss =get_style_loss(transformed_style, style_targets)
    tv_loss = get_total_variation_loss(transformed_img)

    L_style = style_loss * style_weight
    L_content = content_loss * content_weight
    L_tv = tv_loss * tv_weight

    total_loss = L_style + L_content + L_tv
    
    return total_loss
    
def get_content_loss(transformed_outputs, content_targets):
    content_loss = 0
    #assert(len(transformed_outputs) == len(content_outputs))
    for i in transformed_outputs:
        weight = 1
        B, H, W, CH = transformed_outputs[i].get_shape()
        HW = H * W
        loss_i = weight * 2 * tf.nn.l2_loss(transformed_outputs[i]-content_targets[i]) / (B*HW*CH)
        content_loss += loss_i
    return content_loss

def get_style_loss(transformed_outputs, style_targets):
    style_loss = 0
    #assert(len(transformed_outputs) == len(self.S_style_grams))
    for i in transformed_outputs:
        weight = 0.2
        B, H, W, CH = transformed_outputs[i].get_shape()
        G = gram_matrix(transformed_outputs[i])
        A = style_targets[i]
        style_loss += weight * 2 * tf.nn.l2_loss(G - A) / (B * (CH ** 2))
    return style_loss

def get_total_variation_loss(img):
    B, W, H, CH = img.get_shape()
    return tf.reduce_sum(tf.image.total_variation(img)) / (W*H)

def flow_loss(currentStylized_frame, prevStylized_frame):
    currentStylized_frame = tf.cast(currentStylized_frame, tf.float32)
    prevStylized_frame = tf.cast(prevStylized_frame, tf.float32)
    flow_loss = tf.norm((prevStylized_frame - currentStylized_frame)**2, ord='fro', axis=[-3,-2])/(436*1024)
    return flow_loss
