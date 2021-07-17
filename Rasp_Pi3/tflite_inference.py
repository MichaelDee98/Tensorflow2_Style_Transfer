from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import time


def load_img(path_to_img):


    image = cv2.imread(path_to_img)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    input_data = np.expand_dims(image_resized, axis=0)


    return np.float32(input_data)


# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path="./model_temp_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test the model on random input data.
input_shape = input_details[0]['shape']

images = (load_img("./Dataset/000000003553.jpg"), load_img("./Dataset/000000015440.jpg"), load_img("./Dataset/000000007278.jpg"))

for _ in range(4):
    for image in images:
        input_data = image
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        print(f"Elapsed time = {end_time-start_time}")

    

