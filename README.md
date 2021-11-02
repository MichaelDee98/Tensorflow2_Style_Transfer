# Tensorflow 2 Style Transfer[[Johnson et al](https://arxiv.org/abs/1603.08155), [Gupta et al](https://arxiv.org/abs/1705.02092)]
This is a repository for my Diploma thesis on Style Transfer.
These scripts are made for training a Jonhnson et al Image Transform Network and then fine-tuning it with Gupta et al method.

## Transformation Networks
1. **transform_net**

   This is an Image Transform Network that concatenates 2 inputs along the channels, so we can use this for fine tuning afterwords.

2. **transform_netAndroid**

   This is an Image Transform Network with some changes so this network will work with Style Transfer on Android Platforms.

3. **transform_netOnly**

   This is the original Image Transform Network from Johnson et al.

## Modes

1. **train**

   Trains your chosen network.

2. **inference**

   Inference on random pictures from your chosen dataset and computes the average inference time.

3. **inference_android**

   Inference for andoird on a specific picture. This is for testing purposes so we can check that our network works and produce results.

4. **video**

   Inference on video.

5. **fine_tune**

   Fine tuning the chosen Image Transform Network.

6. **android_train**

   Training a model for android platform.
