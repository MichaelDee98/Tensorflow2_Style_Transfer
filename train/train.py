import tensorflow as tf
import numpy as np

import time
import os

import modules.utils as utils
import modules.loss_network as ln



opt = tf.optimizers.Adam(learning_rate=0.001)
EPOCHS = 2


def prepare_dataset(batch_size, style_path, dataset_path):
    print("#### Preparing dataset ####")

    # Load style image
    style_image = utils.load_img(style_path, resize=False)
    print("Style image loaded.")
    # Num of images in the dataset
    content_images_names = os.listdir(dataset_path)
    num_of_images = len(content_images_names)

    # Batch
    batch_shape = (batch_size, 256, 256, 3)
    X_batch = np.zeros(batch_shape, dtype=np.float32)

    train_dataset = tf.data.Dataset.list_files(dataset_path + '*.jpg')
    train_dataset = train_dataset.map(utils.load_img,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print("Train dataset loaded.")
    print("#### Dataset Ready ####")

    return train_dataset, style_image, X_batch, num_of_images




def train_step(X_batch, extractor, training_model, style_targets, content_weight, style_weight, TV_WEIGHT, zeros):
    with tf.GradientTape() as tape:

        content_targets = extractor(X_batch*255.0)['content']
        image = training_model(X_batch, zeros)
        image = tf.clip_by_value(image, 0.0, 255.0)

        outputs = extractor(image, training = True)
        loss = ln.get_loss(outputs['content'], outputs['style'], content_targets, style_targets, image, style_weight, content_weight, TV_WEIGHT)

    grads = tape.gradient(loss, training_model.trainable_variables) 

    opt.apply_gradients(zip(grads, training_model.trainable_variables))
    return loss

def train(model, style_layers, content_layers, batch_size, style_path, dataset_path, content_weight, style_weight, TV_WEIGHT, weights_path):
    print(f"{model}, {style_layers}, {content_layers}, {batch_size}, {style_path}, {dataset_path}, {content_weight}, {style_weight}, {TV_WEIGHT}, {weights_path}")
    zeros = tf.zeros(shape=(batch_size, 256, 256, 3))
    print("Beginning training.")
    training_model = model
    train_dataset, style_image, X_batch, num_of_images = prepare_dataset(batch_size, style_path, dataset_path) 


    extractor = ln.StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image*255)['style']

    


    isBreak = 0
    start = time.time()
    PROGBAR = tf.keras.utils.Progbar(len(train_dataset))
    for n in range(EPOCHS):
        print(f"Epoch = {n + 1}")
        iteration = 0

        for img in train_dataset:

            for j, img_p in enumerate(img):
                X_batch[j] = img_p
            
            iteration +=1
            print(f"{batch_size*iteration}/{num_of_images}")

            loss = train_step(X_batch, extractor, training_model, style_targets, content_weight, style_weight, TV_WEIGHT, zeros)
            print(f"Loss = {loss.numpy()}")

            if iteration%400==0:
                training_model.save_weights(weights_path, save_format="tf")
                print("Weights saved")

            if np.isnan(loss):
                isBreak = 1
                break

            PROGBAR.update(iteration)




        if isBreak:
            break

    print(f"Loss: {loss.numpy()}")
    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    print(f'Network {training_model} finished training')
    print(f'Weights saved on {weights_path}')
    training_model.save_weights(weights_path, save_format="tf")
    print("Final weights saved")
    return training_model


