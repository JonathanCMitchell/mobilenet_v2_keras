# MobileNetV2
This folder contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)


This model file has been pushed to my keras fork which you can see [here](https://github.com/JonathanCMitchell/keras).
You can also view the active pull request to keras [here]
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

to use this model:
```
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input
input_tensor = Input(shape=(224,224, 3)) # or you could put (None, None, 3) for shape
model = MobileNetV2(input_tensor = input_tensor, alpha = 1.0, include_top = True, weights=’imagenet’)

# Now you have a fully loaded model.
```

The model only works with depth_multiplier = 1, although the alpha parameter is able to specify width_multipliers if they are included in [0.35, 0.50, 0.75, 1.0]
Additionally, only square input sizes included in [96, 128, 160, 182, 224] can be used.

The `include_top` parameter can be used to grab the full network, if you set it to false, you will grab the base network before the pooling operation and fully connected layer. 


# Pretrained models
Models can be found [here](https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases)
## Imagenet  Checkpoints

* These results are taken from [tfmobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) but I estimate ours are similar in performance. Except for the Pixel 1 inference time.


## Inference results.
You can grab and load up the pickle file `test_results.p` or you can read the results below:


For questions, comments, and concerns please reach me at jmitchell1991@gmail.com.

```
test_results: [{
    'rows': 224,
    'vector_difference': array([
        [3.06442744e-05, 1.03940765e-05, 6.52904509e-06, ...,
            1.86086560e-04, 1.36749877e-05, 2.29768884e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 1.4,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 0.584200382232666,
    'inference_time_keras': 2.468465566635132,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224.h5',
    'max_vector_difference': 0.0442425,
    'preds_agree': True
}, {
    'rows': 224,
    'vector_difference': array([
        [1.2600726e-04, 1.9243037e-04, 8.6632121e-05, ..., 1.7771832e-05,
            1.2540509e-04, 8.6921602e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 1.3,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 0.7864320278167725,
    'inference_time_keras': 0.48574304580688477,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224.h5',
    'max_vector_difference': 0.1639086,
    'preds_agree': True
}, {
    'rows': 224,
    'vector_difference': array([
        [2.6156631e-06, 8.9272799e-06, 9.9282261e-07, ..., 2.7298967e-05,
            7.0550705e-06, 1.7008846e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 1.0,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 0.9191036224365234,
    'inference_time_keras': 0.5298285484313965,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5',
    'max_vector_difference': 0.008201845,
    'preds_agree': True
}, {
    'rows': 192,
    'vector_difference': array([
        [2.9478622e-05, 3.1182157e-05, 1.6525744e-05, ..., 3.2548003e-05,
            2.3264165e-05, 1.2547210e-04
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 1.0,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 1.0997955799102783,
    'inference_time_keras': 0.609818696975708,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_192.h5',
    'max_vector_difference': 0.03410864,
    'preds_agree': True
}, {
    'rows': 160,
    'vector_difference': array([
        [1.9924228e-06, 1.1813272e-05, 2.6646510e-05, ..., 2.1615942e-06,
            7.3942429e-06, 2.8581535e-06
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 1.0,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 1.3430328369140625,
    'inference_time_keras': 0.6801567077636719,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5',
    'max_vector_difference': 0.0024641072,
    'preds_agree': True
}, {
    'rows': 128,
    'vector_difference': array([
        [1.9651561e-05, 7.6118726e-05, 1.1588483e-05, ..., 2.3098060e-05,
            5.0026181e-05, 2.3178840e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 1.0,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 1.6118056774139404,
    'inference_time_keras': 0.7277970314025879,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128.h5',
    'max_vector_difference': 0.022542655,
    'preds_agree': True
}, {
    'rows': 96,
    'vector_difference': array([
        [4.1969906e-07, 6.2867985e-06, 7.6682009e-06, ..., 9.5812502e-06,
            5.6552781e-06, 2.0793846e-04
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 1.0,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 1.759774923324585,
    'inference_time_keras': 0.8093435764312744,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96.h5',
    'max_vector_difference': 0.009057283,
    'preds_agree': True
}, {
    'rows': 224,
    'vector_difference': array([
        [1.9891046e-05, 3.5158137e-05, 7.4801164e-06, ..., 3.1968251e-05,
            1.8171089e-05, 2.4247890e-04
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.75,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 1.988135576248169,
    'inference_time_keras': 0.907604455947876,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224.h5',
    'max_vector_difference': 0.027664006,
    'preds_agree': True
}, {
    'rows': 192,
    'vector_difference': array([
        [2.4571000e-06, 2.3607154e-06, 1.0745002e-06, ..., 7.6524229e-06,
            2.5452073e-07, 6.0848397e-06
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.75,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 2.1681172847747803,
    'inference_time_keras': 0.9792499542236328,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_192.h5',
    'max_vector_difference': 0.0048509836,
    'preds_agree': True
}, {
    'rows': 160,
    'vector_difference': array([
        [3.3421951e-05, 2.8039271e-05, 2.6024140e-05, ..., 1.8016202e-05,
            2.8351524e-06, 3.2267882e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.75,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 2.4570868015289307,
    'inference_time_keras': 1.0636296272277832,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_160.h5',
    'max_vector_difference': 0.04817468,
    'preds_agree': True
}, {
    'rows': 128,
    'vector_difference': array([
        [5.7490397e-05, 2.3515218e-05, 6.2765699e-05, ..., 6.3877407e-05,
            2.4530049e-05, 1.9020826e-04
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.75,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 2.712172746658325,
    'inference_time_keras': 1.1841137409210205,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_128.h5',
    'max_vector_difference': 0.10443801,
    'preds_agree': True
}, {
    'rows': 96,
    'vector_difference': array([
        [1.91572035e-05, 1.15800685e-05, 4.44647230e-06, ...,
            9.32329567e-06, 4.12614900e-05, 3.88475746e-06
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.75,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 2.947575807571411,
    'inference_time_keras': 1.2835781574249268,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_96.h5',
    'max_vector_difference': 0.02189435,
    'preds_agree': True
}, {
    'rows': 224,
    'vector_difference': array([
        [2.7594273e-05, 1.8192208e-05, 5.0051967e-06, ..., 2.2260952e-05,
            1.0851298e-05, 2.0575267e-04
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.5,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 3.234971761703491,
    'inference_time_keras': 1.4089748859405518,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5',
    'max_vector_difference': 0.03057003,
    'preds_agree': True
}, {
    'rows': 192,
    'vector_difference': array([
        [6.4921463e-05, 1.8974228e-05, 9.5047453e-06, ..., 2.7531139e-05,
            1.5725660e-05, 1.3637640e-04
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.5,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 3.471426010131836,
    'inference_time_keras': 1.5226759910583496,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_192.h5',
    'max_vector_difference': 0.10459131,
    'preds_agree': True
}, {
    'rows': 160,
    'vector_difference': array([
        [2.1799133e-05, 2.6465546e-05, 9.7673910e-07, ..., 4.9333670e-05,
            1.2139077e-05, 3.4930854e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.5,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 3.7173619270324707,
    'inference_time_keras': 1.6358251571655273,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_160.h5',
    'max_vector_difference': 0.041921377,
    'preds_agree': True
}, {
    'rows': 128,
    'vector_difference': array([
        [1.6039543e-05, 3.6521582e-05, 2.0016232e-06, ..., 7.7442382e-06,
            1.3480414e-05, 1.9661791e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.5,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 3.96859073638916,
    'inference_time_keras': 1.7203476428985596,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_128.h5',
    'max_vector_difference': 0.015232503,
    'preds_agree': True
}, {
    'rows': 96,
    'vector_difference': array([
        [5.3964366e-05, 4.3542765e-05, 1.9309173e-05, ..., 1.8680606e-05,
            1.7692482e-05, 1.0907562e-03
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.5,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 4.313321590423584,
    'inference_time_keras': 1.8665125370025635,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_96.h5',
    'max_vector_difference': 0.10974246,
    'preds_agree': True
}, {
    'rows': 224,
    'vector_difference': array([
        [1.8852079e-05, 9.4596544e-07, 2.4118672e-06, ..., 3.3465330e-06,
            3.7501377e-07, 2.0572159e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.35,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 4.690372705459595,
    'inference_time_keras': 1.9682881832122803,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224.h5',
    'max_vector_difference': 0.018127322,
    'preds_agree': True
}, {
    'rows': 192,
    'vector_difference': array([
        [5.89300471e-05, 1.10333494e-05, 2.08540587e-06, ...,
            1.03199854e-04, 1.02247395e-05, 3.36650060e-04
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.35,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 4.746210336685181,
    'inference_time_keras': 2.11893892288208,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_192.h5',
    'max_vector_difference': 0.031285435,
    'preds_agree': True
}, {
    'rows': 160,
    'vector_difference': array([
        [4.2083520e-06, 5.5578494e-07, 3.2600292e-07, ..., 4.2584943e-06,
            5.7042635e-06, 4.6083354e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.35,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 5.078217029571533,
    'inference_time_keras': 2.330106735229492,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_160.h5',
    'max_vector_difference': 0.008356452,
    'preds_agree': True
}, {
    'rows': 128,
    'vector_difference': array([
        [1.0704320e-05, 5.3489948e-06, 7.2533185e-06, ..., 1.6965925e-05,
            2.3451194e-06, 2.8069786e-05
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.35,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 5.38210391998291,
    'inference_time_keras': 2.5224623680114746,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128.h5',
    'max_vector_difference': 0.012936056,
    'preds_agree': True
}, {
    'rows': 96,
    'vector_difference': array([
        [6.1693572e-06, 3.9814022e-06, 3.0157253e-07, ..., 4.4240751e-06,
            2.7236201e-06, 7.0944225e-06
        ]
    ], dtype = float32),
    'pred_keras_score': 389,
    'alpha': 0.35,
    'pred_keras_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'pred_tf_score': 389,
    'inference_time_tf': 5.456079721450806,
    'inference_time_keras': 2.454572916030884,
    'pred_tf_label': 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'model': '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5',
    'max_vector_difference': 0.018446982,
    'preds_agree': True
}]
```