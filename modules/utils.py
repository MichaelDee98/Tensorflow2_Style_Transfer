import tensorflow as tf
import PIL.Image
import numpy as np
import cv2
from random import randrange

def load_img(path_to_img, max_dim=None, resize=True, frame = False):

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

def tensor_to_image(tensor, android = False):
    if android:
        tensor = 255*tensor

    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def clip_0_255(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

def total_variation_loss(img):
    x_var = img[:,:,1:,:] - img[:,:,:-1,:]
    y_var = img[:,1:,:,:] - img[:,:-1,:,:]

    return tf.reduce_mean(tf.square(x_var)) + tf.reduce_mean(tf.square(y_var))

def check_division(pad):
    if(pad%2 == 0):
        add_extra = False
    else:
        add_extra = True

    return add_extra

def load_frame(frame):
    img = cv2.resize(frame,dsize=(256,256), interpolation = cv2.INTER_CUBIC)
    np_image = np.asarray(img)
    img = tf.image.convert_image_dtype(np_image, tf.float32)
    img = img[tf.newaxis, :]
    return img

def GenerateRandomLayer(number_of_layers):
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    model.trainable = False
    # We have not picked any layers yet
    layers = []
    layers_picked = 0
    # Number of layers in model
    no_layers = len(model.layers)

    while layers_picked <  number_of_layers :
        #print(layers_picked)
        # Pick a random layer
        layer_index = randrange(no_layers)
        won_layer = model.layers[layer_index]
        if "_conv" in won_layer.name:
            if won_layer.name in layers:
                continue
            else:
                layers.append(won_layer.name)
                layers_picked = layers_picked + 1

    return layers

def log_information(weights_path, network, content_weight, style_weight, dataset_path, style_path, content_layers, style_layers):
    f = open(weights_path+"information.txt", "w+")

    f.write(f"net: {network} \n")
    f.write(f"content_weight: {content_weight}\n")
    f.write(f"style_weight: {style_weight}\n")
    f.write(f"dataset_path: {dataset_path}\n")
    f.write(f"style_path: {style_path}\n")
    f.write(f"content_layers: {content_layers}\n")
    f.write(f"style_layers: {style_layers}\n")


    f.close()

def read_flo_file(filename, memcached=False):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    if memcached:
        filename = io.BytesIO(filename)
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def flow_wrapper(current_frame, flow):
    #current_frame = np.reshape(current_frame,[1024, 436])
    h, w = flow.shape[:2]
    #flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    prev_wrapped = cv2.remap(np.uint8(current_frame), flow, None, cv2.INTER_LINEAR)
    prev_wrapped = tf.convert_to_tensor(prev_wrapped)
    #print(f"prev_wrapped {prev_wrapped.shape}")
    return prev_wrapped


def flow_loss_data_prep(current_frame, previous_frame, occ_mask, flow):
    """
    This function preprocess the frames for the flow loss function
    """

    #Make occlusion mask uint8 and invert colors
    occ_mask = 1 - occ_mask # invert colors
    tf.cast(occ_mask, tf.uint8) # cast to uint8 
    occ_mask = occ_mask*255 # scale it to 0-255
    occ_mask = tf.squeeze(occ_mask)
    occ_mask = occ_mask.numpy()
    #print(f"Occlusion mask {occ_mask.shape}")

    new_shape = tf.cast([436, 1024], tf.int32)
    current_frame = tf.image.resize(current_frame, new_shape)
    tf.cast(current_frame, tf.uint8) # cast to uint8 
    current_frame = current_frame*255 # scale it to 0-255
    current_frame = tf.squeeze(current_frame)
    current_frame = current_frame.numpy()
    #print(f"Current frame {current_frame.shape}")


    new_shape = tf.cast([436, 1024], tf.int32)
    previous_frame = tf.image.resize(previous_frame, new_shape)
    tf.cast(previous_frame, tf.uint8) # cast to uint8 
    previous_frame = previous_frame*255 # scale it to 0-255
    previous_frame = tf.squeeze(previous_frame)
    previous_frame = previous_frame.numpy()
    #print(f"Previous frame {previous_frame.shape}")
    
    # Wrap the current frame to previous using optical flow
    current_wrapped = flow_wrapper(current_frame, flow)

    prev_masked = np.uint8((previous_frame * occ_mask)/255)
    current_masked = np.uint8((current_wrapped * occ_mask))# removed /255 because current_wrapped and occ_mask are uint8 

    
    return prev_masked, current_masked

def read_files(frame, flo_path, occlusion_masks_path):
    frame_num = int(frame[6:10])
    # Read flow 
    #print(flo_path+"frame_"+str(frame_num-1).zfill(4)+".flo")
    flow = read_flo_file(flo_path+"frame_"+str(frame_num-1).zfill(4)+".flo")
    # Load occlusion mask
    #print(occlusion_masks_path+"frame_"+str(frame_num-1).zfill(4)+".png")
    occ_mask = load_img(occlusion_masks_path+"frame_"+str(frame_num-1).zfill(4)+".png", resize=False)
    return flow , occ_mask
