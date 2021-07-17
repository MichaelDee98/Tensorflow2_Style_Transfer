import tensorflow as tf
import modules.utils as utils

import os
import time
import random


def inference(model_path, image_path, network, android = False):
    model = network
    model.load_weights(model_path)
    zeros = tf.zeros(shape=(1, 256, 256, 3))

    substring = model_path.split("/")
    model_name = substring[-2]

    image = utils.load_img(image_path)

    start_time = time.time()
    output = model(image, zeros)
    end_time = time.time()
    print(f"Elapsed time = {end_time - start_time}")
    if android:
        output =tf.clip_by_value(output, 0.0, 1.0)
        output_image = utils.tensor_to_image(output, android = True)
    else:
        output = utils.clip_0_255(output)
        output_image = utils.tensor_to_image(output)

    output_image.save("/".join(substring[:-1])+"/"+model_name+".jpg")
    #output_image.show()

def rand_multiple_inferences(no_of_inferences, network, dataset_path, andoird = False):
    """
        Function for making multiple inferences to random images.
    """
    img_output = []
    
    # List the images
    content_images_names = os.listdir(dataset_path)
    num_of_images = len(dataset_path) # Number of images
    
    for _ in range(no_of_inferences):
        index = random.randint(1, num_of_images)
        output = network(utils.load_img(dataset_path + content_images_names[index],resize=False))
        if andoird:
            output = tf.clip_by_value(output, 0.0, 1.0)
            img_output.append(utils.tensor_to_image(output, android = True))
        else:
            output = utils.clip_0_255(output)
            img_output.append(utils.tensor_to_image(output))


    return img_output


    