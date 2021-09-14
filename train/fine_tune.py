from numpy.core.fromnumeric import sort
import tensorflow as tf
import numpy as np

import time
import os

import modules.utils as utils
import modules.loss_network as ln



opt = tf.optimizers.Adam(learning_rate=0.001)



def prepare_dataset(style_path, dataset_path, scene):
    print("#### Preparing dataset ####")

    # Load style image
    style_image = utils.load_img(style_path, resize=False)
    print("Style image loaded.")
    # Num of images in the dataset
    content_images_names = os.listdir(dataset_path+"training/clean/"+scene)
    content_images_names = sorted(content_images_names)


    # List flo file
    flo_path = dataset_path+"training/flow/"+scene+"/"
    flo_files = os.listdir(flo_path)
    # List Occlusion Mask
    occlusion_masks_path =dataset_path+"training/occlusions/"+scene+"/"
    occlusion_masks_names = os.listdir(occlusion_masks_path)


    print("Train dataset loaded.")
    print("#### Dataset Ready ####")

    return content_images_names, style_image, flo_path, flo_files, occlusion_masks_path, occlusion_masks_names





def train_step(frames, names, extractor, training_model, zeros, style_targets, content_weight, style_weight, TV_WEIGHT, FLOW_WEIGHT, flo_path, occlusion_masks_path):
    loss =[]
    content_targets = [None, None, None, None]
    stylized_frame = [None, None]
    stylized_frame[0] = training_model(frames[0], zeros, crop=True)

    with tf.GradientTape() as tape:
        for index in range(4):
            content_targets[index] = extractor(frames[index+1]*255.0)['content']
        
        #Time Step 1
        print("Timestep 1")
        inputs = [frames[1],stylized_frame[0]]
        stylized_frame[0] = training_model(inputs[0],inputs[1], crop=True)

        outputs = extractor(stylized_frame[0], training = True)
        loss.append(ln.get_loss(outputs['content'], outputs['style'], content_targets[0], style_targets, stylized_frame[0] , style_weight, content_weight, TV_WEIGHT))
        #Time Step 2
        print("Timestep 2")
        inputs = [frames[2],stylized_frame[0]]
        stylized_frame[1] = training_model(inputs[0],inputs[1], crop=True)

        outputs = extractor(stylized_frame[1], training = True)


        flow , occ_mask = utils.read_files(names[2], flo_path, occlusion_masks_path)
        prev_masked, current_masked = utils.flow_loss_data_prep(stylized_frame[1], stylized_frame[0], occ_mask, flow)
        fl_loss = tf.cast((FLOW_WEIGHT*ln.flow_loss(current_masked, prev_masked)), tf.float32)
        loss.append(ln.get_loss(outputs['content'], outputs['style'], content_targets[1], style_targets, stylized_frame[1] , style_weight, content_weight, TV_WEIGHT)+ fl_loss)

        #Time Step 3
        print("Timestep 3")
        inputs = [frames[3],stylized_frame[1]]
        stylized_frame[0] = training_model(inputs[0],inputs[1], crop=True)

        outputs = extractor(stylized_frame[0], training = True)
        loss.append(ln.get_loss(outputs['content'], outputs['style'], content_targets[2], style_targets, stylized_frame[0] , style_weight, content_weight, TV_WEIGHT))


        #Time Step 4
        print("Timestep 4")
        inputs = [frames[4],stylized_frame[0]]
        stylized_frame[1] = training_model(inputs[0],inputs[1],crop=True)

        outputs = extractor(stylized_frame[1], training = True)


        flow , occ_mask = utils.read_files(names[4], flo_path, occlusion_masks_path)
        prev_masked, current_masked = utils.flow_loss_data_prep(stylized_frame[1], stylized_frame[0], occ_mask, flow)
        fl_loss = tf.cast((FLOW_WEIGHT*ln.flow_loss(current_masked, prev_masked)), tf.float32)
        loss.append(ln.get_loss(outputs['content'], outputs['style'], content_targets[3], style_targets, stylized_frame[1] , style_weight, content_weight, TV_WEIGHT)+ fl_loss)



        overal_loss = loss[0] + loss[1] + loss[2] + loss[3]


    grads = tape.gradient(overal_loss, training_model.trainable_variables) 

    opt.apply_gradients(zip(grads, training_model.trainable_variables))
    return loss

def train(model, model_path, style_layers, content_layers, style_path, dataset_path, content_weight, style_weight, TV_WEIGHT, weights_path, FLOW_WEIGHT):
    print("Beginning training.")
    training_model = model
    training_model.load_weights(model_path).expect_partial()
    zeros = tf.zeros(shape=(1, 218, 512, 3))


    scenes = os.listdir(dataset_path+"training/clean/")
    epochs = 10
    
    for ep in range(epochs):
        num_scene = 0
        print(f"Epoch {ep+1}")
        for scene in scenes:
            num_scene += 1
            print(f"Loading scene {scene}")

            content_images_names, style_image, flo_path, flo_files, occlusion_masks_path, occlusion_masks_names = prepare_dataset(style_path, dataset_path, scene) 


            extractor = ln.StyleContentModel(style_layers, content_layers)
            style_targets = extractor(style_image*255)['style']

            

            
            no_frames = len(os.listdir(dataset_path+"training/clean/"+scene+"/"))
            iterations = tf.floor(no_frames/5).numpy()
            iterations = iterations.astype(int)

            start = time.time()
            frames = [None, None, None, None, None]
            names = [None, None, None, None, None]

            for n in range(iterations):
                

                iteration = 1
                print("Loading Frames...")
                for index in range(5):
                    names[index] = content_images_names[(4*n)+index]
                    print(names[index])
                    frames[index] = utils.load_img(dataset_path+"training/clean/"+scene+"/"+content_images_names[(4*n)+index], frame=True)
                

                print("Training...")
                print(f"Epoch {ep+1} Scene {num_scene}/{len(scenes)}")
                loss = train_step(frames, names, extractor, training_model, zeros, style_targets, content_weight, style_weight, TV_WEIGHT, FLOW_WEIGHT, flo_path, occlusion_masks_path)


            end = time.time()
            print("Total time: {:.1f}".format(end-start))
            print(f'Network {training_model} finished training')
            print(f'Weights saved on {weights_path}')
            training_model.save_weights(weights_path, save_format="tf")
            print("Final weights saved")
    


