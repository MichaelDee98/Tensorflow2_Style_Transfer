import tensorflow as tf
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



import modules.utils as utils
import modules.transformation_networks as tn
import train.train_android as train_android
import train.train as train
import train.fine_tune as fine_tune
import inference.inference as inference
import video.video as video
net_dictionary = {
    "transform_net"        : tn.TransformNet(),
    "transform_netAndroid" :tn.TransformNetAndroid()
}


########GLOBAL VARIABLES########


MODE = "train" #the available modes are train, inference, video, video_davis, fine_tune, to_model, android_train
STYLE_PATH = "style_images/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
DATASET_PATH = "./dataset/val2017/" #Dataset path for content images
SAVED_WEIGHTS_PATH ="./saved_weights/"
TRANS_NETWORK = "transform_net"

BATCH = 10
CONTENT_WEIGHT =1e0
STYLE_WEIGHT = 4e1
TV_WEIGHT = 2e2
FLOW_WEIGHT = 100

#Commong content & style layers
CONTENT_LAYERS = ['block4_conv2']
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
################################


#String to bool function for arg pass
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def arg_pass():
    parser = argparse.ArgumentParser(description='Algorithm for training a style transfer network')
    
    parser.add_argument('--mode', required=False, default=MODE)
    parser.add_argument('--weights_path', required=False , default=SAVED_WEIGHTS_PATH,
                        help="Enter the path that you want to save the weights e.g /home/user/Desktop/saved_weights/") 
    parser.add_argument('--style_path', required=False, default=STYLE_PATH,
                        help="Enter the style image path e.g /home/user/Desktop/pics/starry_night.jpg")
    parser.add_argument('--model_path', required=False, default= None)              
    parser.add_argument('--dataset_path', required=False, default=DATASET_PATH, 
                        help="Enter the dataset_path e.g /home/user/Desktop/dataset/val2017")
    parser.add_argument('--network', required=False, default=TRANS_NETWORK,
                        help="Choose the network you want to train. Valid nets are transform_net. Default is transform_net")
    parser.add_argument('--batch', required=False,type=int, default = BATCH,
                        help="Choose batch size for training. Default is 10")
    parser.add_argument('--content_weight', required=False, type=int, default=CONTENT_WEIGHT,
                        help='Content Weight. Default is 1e0')
    parser.add_argument('--style_weight', required=False, type=float, default=STYLE_WEIGHT,
                        help='Style Weight. Default is 4e1.')
    parser.add_argument('--video', required=False,
                    help="Choose the video you want to style.")
    parser.add_argument('--style_it', required=False, type = str2bool)



    args = parser.parse_args()

    mode = args.mode
    weights_path = args.weights_path
    style_path = args.style_path
    model_path = args.model_path
    dataset_path = args.dataset_path
    network = args.network
    batch = args.batch
    content_weight = args.content_weight
    style_weight = args.style_weight
    video = args.video
    style_it = args.style_it
    

    return mode, weights_path, style_path, model_path, dataset_path, network, batch, content_weight, style_weight, video, style_it



#Main of script
if __name__ == "__main__":
    mode, weights_path, style_path, model_path, dataset_path, network, batch, content_weight, style_weight, video_path, style_it = arg_pass()
    model = net_dictionary[network]

    if(mode=="train"):
        
        weights_file = weights_path
        # Creating a new directory for saving the model weights
        folder_list = os.listdir(weights_path)
        for element in folder_list:
            if "." in element:
                folder_list.remove(element)

        content_layers = CONTENT_LAYERS

        style_layers = STYLE_LAYERS
        
        os.mkdir(weights_path+"model_"+str(len(folder_list)+1)+"/")
        
        utils.log_information(weights_file+"model_"+str(len(folder_list)+1)+"/", network,content_weight, style_weight, dataset_path, style_path, content_layers, style_layers)
        weights_path = weights_file+"model_"+str(len(folder_list)+1)+"/model"

        
        trained_model = train.train(model, style_layers, content_layers, batch, style_path, dataset_path, content_weight, style_weight, TV_WEIGHT , weights_path)
        print("Inference on sample. Will produce 10 images")

    

        substring = weights_path.split("/")
        model_name = substring[-2]

    
        outputs = inference.rand_multiple_inferences(10, model, dataset_path)
        
        for i, output in enumerate(outputs):
            output.save("/".join(substring[:-1])+"/"+model_name+"rand"+str(i)+".jpg")

    elif(mode=="inference"):
        assert model_path is not None
        
        #model = net_dictionary[network]
        #substring = weights_path.split("/")
        #model_name = substring[-2]

        #inference.inference(model_path,"./dataset/mini_batch/000000365766.jpg", model)
        metrics = (inference.rand_multiple_inferences(100, model, dataset_path))
        print(f"Average of metrics {tf.reduce_mean(metrics)}")

        #for i, output in enumerate(outputs):
        #    output.save("/".join(substring[:-1])+"/"+model_name+"rand"+str(i)+".jpg")

    elif(mode=="inference_android"):
        assert model_path is not None
        model = net_dictionary["transform_netAndroid"]
        #substring = weights_path.split("/")
        #model_name = substring[-2]
        inference.inference(model_path,"./dataset/mini_batch/000000365766.jpg", model, android=True)
        #outputs = inference.rand_multiple_inferences(10, model, dataset_path)
        #for i, output in enumerate(outputs):
        #    output.save("/".join(substring[:-1])+"/"+model_name+"rand"+str(i)+".jpg")
    elif(mode=="video"):
        assert model_path is not None
        assert video_path is not None

        video.load_video(video_path, model_path, model)
    elif(mode=="video_davis"):
        assert model_path is not None
        assert dataset_path is not None
        assert style_it is not None

        video.make_video(dataset_path, model, model_path, style_it)
    elif mode=="fine_tune":


        weights_file = weights_path
        # Creating a new directory for saving the model weights
        folder_list = os.listdir(weights_path)
        for element in folder_list:
            if "." in element:
                folder_list.remove(element)

        content_layers = CONTENT_LAYERS

        style_layers = STYLE_LAYERS
        
        os.mkdir(weights_path+"Agrim_model_"+str(len(folder_list)+1)+"/")
        
        utils.log_information(weights_file+"Agrim_model_"+str(len(folder_list)+1)+"/", network,content_weight, style_weight, dataset_path, style_path, content_layers, style_layers)
        weights_path = weights_file+"Agrim_model_"+str(len(folder_list)+1)+"/model"


        fine_tune.train(model, model_path, style_layers, content_layers, style_path, dataset_path, content_weight, style_weight, TV_WEIGHT , weights_path, FLOW_WEIGHT)
    elif mode=="to_model":
        model.load_weights(model_path).expect_partial()
        inputs = tf.keras.Input(shape=[256, 256, 3], batch_size= 1)
        outputs = model(inputs)
        savedModel = tf.keras.Model(inputs, outputs)
        savedModel.save("saved_model/temp_model")
    elif mode=="android_train":
        model = net_dictionary["transform_netAndroid"]
        weights_file = weights_path
        # Creating a new directory for saving the model weights
        folder_list = os.listdir(weights_path)
        for element in folder_list:
            if "." in element:
                folder_list.remove(element)

        content_layers = CONTENT_LAYERS

        style_layers = STYLE_LAYERS
        
        os.mkdir(weights_path+"model_"+str(len(folder_list)+1)+"/")
        
        utils.log_information(weights_file+"model_"+str(len(folder_list)+1)+"/", network,content_weight, style_weight, dataset_path, style_path, content_layers, style_layers)
        weights_path = weights_file+"model_"+str(len(folder_list)+1)+"/model"

        
        trained_model = train_android.train(model, style_layers, content_layers, batch, style_path, dataset_path, content_weight, style_weight, TV_WEIGHT , weights_path)
        print("Inference on sample.")

    

        substring = weights_path.split("/")
        model_name = substring[-2]

    
        outputs = inference.rand_multiple_inferences(10, model, dataset_path, andoird=True)
        for i, output in enumerate(outputs):
            output.save("/".join(substring[:-1])+"/"+model_name+"rand"+str(i)+".jpg")