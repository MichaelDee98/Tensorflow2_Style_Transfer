import tensorflow as tf
import time


from modules.utils import load_img, tensor_to_image


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./model_temp_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test the model on random input data.
input_shape = input_details[0]['shape']
images = (load_img("./dataset/mini_batch/000000446117.jpg"), load_img("./dataset/mini_batch/000000000802.jpg"))


for image in images:
    input_data = image
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()
    print(f"Elapsed time = {end_time-start_time}")
    print(tf.reduce_max(output_data))
    image = tensor_to_image(output_data)
    
    image.show()