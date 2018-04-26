"""MobileNet v1 models for Keras.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).

The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------

The weights for all 16 models are obtained and translated
from TensorFlow checkpoints found at
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import h5py

# TODO: Change these to be from .. imports
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Add
from keras.layers import Dense
from keras.layers import DepthwiseConv2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
# from keras.engine import get_source_inputs
# from keras.engine.base_layer import InputSpec
from keras.engine import InputSpec
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

# TODO Change these to real path
WEIGHTS_PATH = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v0.1/keras_mobilenet_1_224_weights.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v0.1/keras_mobilenet_1_224_weights_no_top.h5'

def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    This function applies the "Inception" preprocessing which converts
    the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
    function is different from `imagenet_utils.preprocess_input()`.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    x /= 128.
    x -= 1.
    return x.astype(np.float32)

def MobileNetv2(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3, # not in use
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1001): # we include the background class
    """
    Instantiates the MobileNet architecture.

    To load a MobileNet model via `load_model`, import the custom
    objects `relu6` and pass them to the `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6})

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution
            (also called the resolution multiplier)
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1001:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1001')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            if rows is None:
                rows = 224
                warnings.warn('MobileNet shape is undefined.'
                              ' Weights for input shape (224, 224) will be loaded.')
            else:
                raise ValueError('If imagenet weights are being loaded, '
                                 'input must have a static square shape (one of '
                                 '(128, 128), (160, 160), (192, 192), or (224, 224)).'
                                 ' Input shape provided = %s' % (input_shape,))

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    

    x = Conv2D(32, kernel_size=3, strides=(2, 2), padding='same',
               use_bias=False, name='Conv1')(img_input)
    x = BatchNormalization(epsilon=1e-5, name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)

    # x = _inverted_res_block(x, filters=16, stride=1, expansion=1, block_id=0) # there are no expansion weights here
    x = _first_inverted_res_block(
        x, filters=16, stride=1, expansion=1, block_id=0)
    x = _inverted_res_block(x, filters=24, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, stride=1,
                            expansion=6, block_id=2)  # c2

    x = _inverted_res_block(x, filters=32, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, stride=1,
                            expansion=6, block_id=5)  # c3

    x = _inverted_res_block(x, filters=64, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, stride=1,
                            expansion=6, block_id=12)  # c4

    x = _inverted_res_block(x, filters=160, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, stride=1,
                            expansion=6, block_id=15)  # c5

    x = _inverted_res_block(x, filters=320, stride=1,
                            expansion=6, block_id=16)  # c6
    # TODO Can take C6 and overwrite function

    x = Conv2D(1280, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-5, name='Conv_1_bn')(x)
    x = Activation(relu6)(x)

    if include_top:
        # TODO Add logic for if K.image_data_format() == 'channels_first' to work with non tf models
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(classes, activation='softmax',
                  use_bias=True, name='Logits')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_first" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        # TODO include other alpha models
        # elif alpha == 0.75:
        #     alpha_text = '7_5'
        # elif alpha == 0.50:
        #     alpha_text = '5_0'
        # else:
        #     alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenetv2_%s_%d_tf.h5' % (alpha_text, rows)
            weights_path = get_file(model_name,
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            model_name = 'mobilenetv2_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weights_path = get_file(model_name,
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def _inverted_res_block(inputs, expansion, stride, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    # Expand
    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None,
               name='mobl%d_conv_%d_expand' % (block_id, block_id))(inputs)
    x = BatchNormalization(epsilon=1e-5, name='bn%d_conv_%d_bn_expand' %
                           (block_id, block_id))(x)
    x = Activation(relu6, name='conv_%d_relu' % block_id)(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same',
                        name='mobl%d_conv_%d_depthwise' % (block_id, block_id))(x)
    x = BatchNormalization(epsilon=1e-5, name='bn%d_conv_%d_bn_depthwise' %
                           (block_id, block_id))(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # Project
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False,
               activation=None, name='mobl%d_conv_%d_project' % (block_id, block_id))(x)
    x = BatchNormalization(epsilon=1e-5, name='bn%d_conv_%d_bn_project' %
                           (block_id, block_id))(x)

    if in_channels == filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def _first_inverted_res_block(inputs, expansion, stride, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same',
                        name='mobl%d_conv_%d_depthwise' % (block_id, block_id))(inputs)
    x = BatchNormalization(epsilon=1e-5, name='bn%d_conv_%d_bn_depthwise' %
                           (block_id, block_id))(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # Project
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False,
               activation=None, name='mobl%d_conv_%d_project' % (block_id, block_id))(x)
    x = BatchNormalization(epsilon=1e-5, name='bn%d_conv_%d_bn_project' %
                           (block_id, block_id))(x)

    if in_channels == filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x
