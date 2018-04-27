# MobileNetV2
This folder contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

# Performance
## Latency
This is the timing of [MobileNetV1](../mobilenet_v1.md) vs MobileNetV2 using
TF-Lite on the large core of Pixel 1 phone.

![mnet_v1_vs_v2_pixel1_latency.png](mnet_v1_vs_v2_pixel1_latency.png)

# This model checkpoint was downloaded from the following source:
| [mobilenet_v2_1.0_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) | 300 | 3.47 | 71.8 | 91.0 | 73.8


First, I chose to extract all the weights from Tensorflows [repo](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) and save the depth_multiplier values and input resolutions to a file `models_to_load.py`. 
The for each model in `models_to_load.py`, I extracted the weights using file `extract_weights.py`, utilizing the checkpoints provided,  and saved the weights to a directory called 'weights'. Then I used the file `load_weights_multiple.py`  to set the weights of the corresponding keras model using keras's built in `set_weights` function. I used a pickle file that was generated using `extract_weights.py` to serve as a guide and provide meta data about each layer so that I could align them. Each weight is checked for:
Shape, mod (expand, depthwise, or project), meta: (weights or batch norm parameters), and size. 
### The model is then tested inside `test_mobilenet.py`. This model is tested against the tensorflow slim model that can be found [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)


