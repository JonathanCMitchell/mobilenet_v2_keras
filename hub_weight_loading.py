import numpy as np
import tensorflow as tf
import tensornets as nets
import tensorflow_hub as hub
# from mobilenetv2 import MobileNetV2
from keras.models import Model
from keras.applications.mobilenetv2 import MobileNetV2
# from mobilenetv2 import MobileNetV2


def map_alpha_to_slim(alpha):
    alpha_map = {
        1.4: '140',
        1.3: '130',
        1.0: '100',
        0.75: '075',
        0.5: '050',
        0.35: '035'
    }
    return alpha_map[alpha]


alpha = 0.35
rows = 192
# rows = 224
# rows = 160
# rows = 128
# rows = 96

print('ALPHA: ', alpha)
print('rows:', rows)

WEIGHTS_SAVE_PATH_INCLUDE_TOP = '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + \
    str(alpha) + '_' + str(rows) + '.h5'

WEIGHTS_SAVE_PATH_NO_TOP = '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + \
    str(alpha) + '_' + str(rows) + '_no_top' + '.h5'

img = nets.utils.load_img('cat.png', target_size=256, crop_size=rows)
img = (img / 128.0) - 1.0

inputs = tf.placeholder(tf.float32, [None, rows, rows, 3])
model = hub.Module(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_" + map_alpha_to_slim(alpha) + "_" + str(rows) + "/classification/1")
features = model(inputs, signature="image_classification", as_dict=True)
probs = tf.nn.softmax(features['default'])

# with tf.variable_scope('keras'):
print('for ALPHA: ', alpha)

model2 = MobileNetV2(weights='imagenet', alpha = alpha, input_shape = (rows, rows, 3))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    preds = sess.run(probs, {inputs: img})

preds2 = model2.predict(img)

print('TFHUB: ', nets.utils.decode_predictions(preds[:, 1:]))
print('MOBLV2 local bn new: ',nets.utils.decode_predictions(preds2))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    weights = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='module/MobilenetV2')
    values = sess.run(weights)
    values[-2] = np.delete(np.squeeze(values[-2]), 0, axis=-1)
    values[-1] = np.delete(values[-1], 0, axis=-1)

model2.set_weights(values)

# Save weights no top and model
model2.save_weights(WEIGHTS_SAVE_PATH_INCLUDE_TOP)
model2_no_top = Model(input=model2.input,
                        output=model2.get_layer('out_relu').output)
model2_no_top.save_weights(WEIGHTS_SAVE_PATH_NO_TOP)


preds3 = model2.predict(img)



print('MOBLV2 local bn new weights new: ', nets.utils.decode_predictions(preds3))


# Now try to load new model locally and get the same weight score. 

