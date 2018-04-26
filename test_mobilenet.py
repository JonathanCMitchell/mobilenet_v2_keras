# Test script to show that the mobilenetV2 works as expected
from __future__ import print_function
from keras.layers import Input
from keras.utils import get_file
import numpy as np
# from mobilenetv2 import MobileNetV2
from mobilenetv2 import MobileNetV2
import urllib
import json
import PIL
import time
from imagenet_utils import create_readable_names_for_imagenet_labels
import tensorflow as tf
import gzip
import tarfile
import os
import sys
# PYTHONPATH should contain the research/slim/ directory in the tensorflow/models repo.
from nets.mobilenet import mobilenet_v2

from keras.models import Model

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# ==== Load MobilenetV2 Keras Version ====
# ========================================
def predict_keras(img, alpha, rows, weights_path):
    """
    params: img: an input image with shape (1, 224, 224, 3)
            note: Image has been preprocessed (x /= 127.5 - 1)
    Runs forward pass on network and returns logits and the inference time
    """
    input_tensor = Input(shape=(224, 224, 3))
    
    # model = MobileNetv2(input_tensor=input_tensor, include_top=True, weights='imagenet')

    model = MobileNetV2(input_tensor=input_tensor,
                        include_top=True, weights=None, alpha = alpha)
    model.load_weights(weights_path)



    tic = time.time()
    output_logits = model.predict(img)
    toc = time.time()
    return output_logits, toc-tic

def get_tf_mobilenet_v2_items(alpha, rows):
    model_path = os.path.join(MODEL_DIR, 'mobilenet_v2_' + str(alpha) + '_' + str(rows))
    base_name = 'mobilenet_v2_' + str(alpha) + '_' + str(rows)
    base_path = os.path.join(model_path, base_name)

    url = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/' + base_name + '.tgz'
    print('Downloading from ', url)
    
    urllib.request.urlretrieve(url, model_path + '.tgz')
    tar = tarfile.open(model_path + '.tgz', "r:gz")
    tar.extractall(model_path)
    tar.close()

    return base_path
    

def predict_slim(img, checkpoint, rows):
    """
    params: img: a preprocessed image with shape (1, 224, 224, 3)
            checkpoint: the path to the frozen.pb checkpoint
    Runs a forward pass of the tensorflow slim mobilenetV2 model which has been frozen for inference
    returns: numpy array x, which are the logits, and the inference time
    """
    gd = tf.GraphDef.FromString(open(checkpoint + '_frozen.pb', 'rb').read())
    inp, predictions = tf.import_graph_def(
    gd,  return_elements=['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])

    with tf.Session(graph=inp.graph):
        tic = time.time()
        x = predictions.eval(feed_dict={inp: img.reshape(1, rows,rows, 3)})
        toc = time.time()
    return x, toc-tic


if __name__ == "__main__":

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)


    alpha = 0.5
    rows = 224

    WEIGHTS_SAVE_PATH_INCLUDE_TOP = '/home/jon/Documents/keras_mobilenetV2/test_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + \
        str(alpha) + '_' + str(rows) + '.h5'

   
    # Get tensorflow checkpoint path and download required items
    SLIM_CKPT_base_path = get_tf_mobilenet_v2_items(alpha = alpha, rows = rows)

    # To test Panda image
    url = 'https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG'
    img_filename = 'panda.jpg'

    # To test Monkey image
    # url = 'https://upload.wikimedia.org/wikipedia/commons/a/a9/Macaca_sinica_-_01.jpg'
    # img_filename = 'monkey.jpg'

    # Grab test image
    urllib.request.urlretrieve(url, img_filename)

    # Preprocess
    img = np.array(PIL.Image.open(img_filename).resize(
        (rows, rows))).astype(np.float) / 128 - 1
    img = np.expand_dims(img, axis=0)

    # Keras model test
    output_logits_keras, tk = predict_keras(img, alpha = alpha, rows = rows, weights_path = WEIGHTS_SAVE_PATH_INCLUDE_TOP)

    # Tensorflow SLIM
    output_logits_tf, tt = predict_slim(img, SLIM_CKPT_base_path, rows)

    label_map = create_readable_names_for_imagenet_labels()
    print("Prediction keras: ", output_logits_keras.argmax(),label_map[output_logits_keras.argmax()], "score: ", output_logits_keras.max())
    print("Prediction tf: ", output_logits_tf.argmax(), label_map[output_logits_tf.argmax()], "score: " , output_logits_tf.max())
    print("Inference time keras: ", tk)
    print("Inference time tf: ", tt)
    print('Output logits deviation: ', np.allclose(output_logits_keras, output_logits_tf, 0.5))
