import tensorflow as tf
import modules.utils as utils
import Adain.adain as adain



encoder = tf.keras.applications.VGG19(include_top=False)
decoder = adain.get_decoder()
model = adain.Net(encoder, decoder)



model.load_weights("/home/michlist/Desktop/Style_Transfer/github/johnson_style_transferV2/saved_weights/Adain/ckpts/ckpt-2")

output = model("./dataset/mini_batch/000000365766.jpg")
print("OUTPUT \n")
print(output)
utils.tensor_to_image(output)
