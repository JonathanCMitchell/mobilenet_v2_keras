# Test script to show that the mobilenetV2 works as expected
from __future__ import print_function
from keras.layers import Input
from keras.utils import get_file
import numpy as np
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
from models_to_load import models_to_load
from keras.models import Model
import pickle

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
    input_tensor = Input(shape=(rows, rows, 3))
    
    # TODO Insert this line once we re-insert formerly deleted model files
    # model = MobileNetv2(input_tensor=input_tensor, include_top=True, weights='imagenet')

    model = MobileNetV2(input_tensor=input_tensor,
                        include_top=True, weights='imagenet', alpha = alpha)




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

    # maybe reshaping image twice img.reshape(1, rows,rows, 3)
    with tf.Session(graph=inp.graph):
        tic = time.time()
        x = predictions.eval(feed_dict={inp: img})
        toc = time.time()
    return x, toc-tic

def test_keras_and_tf(models = []):
    # This test runs through all the models included below and tests them
    results = []
    for alpha, rows in models:
        # TODO replace WEIGHTS_SAVE_PATH_INCLUDE_TOP with repo link
        WEIGHTS_SAVE_PATH_INCLUDE_TOP = '/home/jon/Documents/keras_mobilenetV2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + \
            str(alpha) + '_' + str(rows) + '.h5'

        # Get tensorflow checkpoint path and download required items
        SLIM_CKPT_base_path = get_tf_mobilenet_v2_items(alpha=alpha, rows=rows)

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
        output_logits_keras, tk = predict_keras(
            img, alpha=alpha, rows=rows, weights_path=WEIGHTS_SAVE_PATH_INCLUDE_TOP)

        # Tensorflow SLIM
        output_logits_tf, tt = predict_slim(img, SLIM_CKPT_base_path, rows)

        label_map = create_readable_names_for_imagenet_labels()
        print('for Model: alpha: ', alpha, "rows: ", rows)
        print("Prediction keras: ", output_logits_keras.argmax(
        ), label_map[output_logits_keras.argmax()], "score: ", output_logits_keras.max())
        print("Prediction tf: ", output_logits_tf.argmax(),
            label_map[output_logits_tf.argmax()], "score: ", output_logits_tf.max())
        print("Inference time keras: ", tk)
        print("Inference time tf: ", tt)
        print('Output logits deviation: ', np.allclose(
            output_logits_keras, output_logits_tf, 0.5))
        assert(output_logits_tf.argmax() == output_logits_keras.argmax())
        results.append({
            "model": WEIGHTS_SAVE_PATH_INCLUDE_TOP,
            "alpha": alpha,
            "rows": rows,
            "pred_keras_score": output_logits_keras.argmax(),
            "pred_keras_label": label_map[output_logits_keras.argmax()],
            "inference_time_keras": tk,
            "pred_tf_score": output_logits_tf.argmax(),
            "pred_tf_label": label_map[output_logits_tf.argmax()],
            "inference_time_tf": tt,
            "preds_agree": output_logits_keras.argmax() == output_logits_tf.argmax(),
            "vector_difference": np.abs(output_logits_keras - output_logits_tf),
            "max_vector_difference": np.abs(output_logits_keras - output_logits_tf).max(),
        })
    return results
        



if __name__ == "__main__":

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    test_results = test_keras_and_tf(models=models_to_load)

    with open('test_results_2.p', 'wb') as pickle_file:
        pickle.dump(test_results, pickle_file)


    print('test_results: ', test_results)
    
