import numpy as np
import tensorflow as tf
import tensornets as nets
import tensorflow_hub as hub
from mobilenetv2 import MobileNetV2
from keras.models import Model
from keras.layers import Input
from models_to_load import models_to_load


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

def load_hub_weights(models):
    for alpha, rows in models:

        tf.reset_default_graph()
        print('alpha: ', alpha, 'rows: ', rows)

        WEIGHTS_SAVE_PATH_INCLUDE_TOP = '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(rows) + '.h5'

        WEIGHTS_SAVE_PATH_NO_TOP = '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + \
            str(alpha) + '_' + str(rows) + '_no_top' + '.h5'

        # Load tf stuff
        img = nets.utils.load_img('cat.png', target_size=256, crop_size=rows)
        img = (img / 128.0) - 1.0
        inputs = tf.placeholder(tf.float32, [None, rows, rows, 3])

        model = hub.Module(
            "https://tfhub.dev/google/imagenet/mobilenet_v2_"
            + map_alpha_to_slim(alpha) + "_"
            + str(rows) + "/classification/1")

        h, w = hub.get_expected_image_size(model)

        features = model(inputs, signature="image_classification", as_dict=True)
        probs = tf.nn.softmax(features['default'])

        # Load local model
        with tf.variable_scope('keras'):
            model2 = MobileNetV2(weights=None, 
                                 alpha = alpha, 
                                 input_shape=(rows, rows, 3))
            model2.load_weights('./old_weights_nonhub/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) +  '_' +str(rows) + '.h5')
        
        preds1 = model2.predict(img)
        print('preds1: (remote weights) new BN no set w:: ',
              nets.utils.decode_predictions(preds1))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            weights = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='module/MobilenetV2')
            values = sess.run(weights)
            values[-2] = np.delete(np.squeeze(values[-2]), 0, axis=-1)
            values[-1] = np.delete(values[-1], 0, axis=-1)
            sess.close()

        # Save weights no top and model
        model2.set_weights(values)
        model2.save_weights(WEIGHTS_SAVE_PATH_INCLUDE_TOP)
        model2_no_top = Model(input = model2.input, output = model2.get_layer('out_relu').output)
        model2_no_top.save_weights(WEIGHTS_SAVE_PATH_NO_TOP)

        # Predictions with new BN, new weights
        preds2 = model2.predict(img)

        print('preds2: (after set weights) ',
              nets.utils.decode_predictions(preds2))


if __name__ == "__main__":
    # Note: if you want to load and save all the models, you have to reset the tf graph and tf session
    load_hub_weights(models=[(1.0, 224)]])
