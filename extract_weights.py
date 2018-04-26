from __future__ import print_function
import numpy as np
import urllib
from nets.mobilenet import mobilenet_v2
import tensorflow as tf
import gzip
import tarfile
from test_mobilenet import get_tf_mobilenet_v2_items
import sys
import os
import pickle
from models_to_load import models_to_load
# PYTHONPATH should contain the research/slim/ directory in the tensorflow/models repo.

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

def extract_weights(models = []):

    for alpha, rows in models:

        SLIM_CKPT_base_path = get_tf_mobilenet_v2_items(alpha=alpha, rows=rows)

        checkpoint = SLIM_CKPT_base_path + '.ckpt'

        reader = tf.train.NewCheckpointReader(SLIM_CKPT_base_path + '.ckpt')

        # Get checkpoint and then do the rest

        # Obtain expanded keys and not_expanded keys
        tensor_count = 0
        weights_count = 0
        project_count = 0
        expand_count = 0
        depthwise_count = 0
        key_not_exp = 0
        layer_kind_count = {}
        expanded_keys = []
        not_expanded_keys = []
        for key in reader.get_variable_to_shape_map():
            if key.split('/')[-1] == 'ExponentialMovingAverage':
                continue
            if key.split('/')[-1] == 'RMSProp' or key.split('/')[-1] == 'RMPSProp_1':
                continue
            if key == 'global_step':
                continue

            if 'expanded' not in key.split('/')[1]:
                key_not_exp += 1
                not_expanded_keys.append(key)
            else:
                expanded_keys.append(key)

            base = key.split('/')[0]
            block_id = key.split('/')[1]
            layer_kind = key.split('/')[2]
            T = reader.get_tensor(key)

            tensor_count += 1

        # Handle not_expanded keys:
        # add shapes and filter out RMSProp to non expanded keys
        not_expanded_layers = []
        for key in not_expanded_keys:
            if key.split('/')[-1] == 'RMSProp_1':
                continue
            if len(key.split('/')) == 4:
                _, layer, kind, meta = key.split('/')
            elif len(key.split('/')) == 3:
                _, layer, meta = key.split('/')

            if layer == 'Conv':
                block_id = -1
                layer = 'Conv1'
                if meta in ['gamma', 'moving_mean', 'moving_variance', 'beta']:
                    layer = 'bn_Conv1'

            elif layer == 'Conv_1':
                block_id = 17
                if meta in ['gamma', 'moving_mean', 'moving_variance', 'beta']:
                    layer = 'Conv_1_bn'
            elif layer == 'Logits':
                block_id = 19
            else:
                print('something odd encountered')
                continue

            not_expanded_layers.append({
                'key': key,
                'block_id': block_id,
                'layer': layer,
                'mod': '',
                'meta': meta,
                'shape': reader.get_tensor(key).shape,
            })

        # Perform analysis on expanded keys
        expanded_weights_keys = []
        expanded_bn_keys = []
        for key in expanded_keys:
            # if it's length = 5 then it should be a batch norm
            # if it's len = 4 then its a conv
            if len(key.split('/')) == 4:
                #         print('weights keys: ', key)
                T = reader.get_tensor(key)
                expanded_weights_keys.append((key, T.shape))
            elif len(key.split('/')) == 5:
                #         print('batchnorm keys: ', key)
                T = reader.get_tensor(key)
                expanded_bn_keys.append((key, T.shape))
            else:
                # otherwise it's a gamma/RMSProp key
                continue


        print('len of expanded_weights keys: ', len(expanded_weights_keys))
        print('len of expanded_bn_keys: ', len(expanded_bn_keys))


        # Layer will be
        # Block_id = -1 layer => 'Conv' 'bn_Conv_1'
        # Block_id = 17 layer => 'Conv_1' 'Conv_1_bn'
        # Block_id = 19 layer => 'logits', this is a Dense layer

        # Loop through expanded weights keys and create guide
        bn_layer_guide = []
        count = 0
        for bnkey, bnshape in expanded_bn_keys:

            #     # save the file
            _, layer, mod, kind, meta = bnkey.split('/')
            if kind == 'BatchNorm':

                if layer == 'expanded_conv':
                    # then layer is depthwise
                    bn_layer_guide.append({
                        "key": bnkey,
                        'block_id': 0,
                        'layer': kind,
                        'mod': mod,
                        'meta': meta,
                        'shape': bnshape
                    })
                else:
                    num = layer.split('_')[-1]
                    # then layer is depthwise
                    bn_layer_guide.append({
                        "key": bnkey,
                        'block_id': num,
                        'layer': kind,
                        'mod': mod,
                        'meta': meta,
                        'shape': bnshape
                    })

        # Loop through expanded weights keys and create guide
        w_layer_guide = []
        for wkey, wshape in expanded_weights_keys:

            # save the file
            _, layer, mod, meta = wkey.split('/')
            if len(layer.split('_')) == 2:
                # This is expanded_conv_0

                if mod == 'depthwise':
                    kind = 'DepthwiseConv2D'
                elif mod == 'expand' or mod == 'project':
                    kind = 'Conv2D'

                    # then layer is depthwise
                w_layer_guide.append({
                    "key": wkey,
                    'block_id': 0,
                    'layer': kind,
                    'mod': mod,
                    'meta': meta,
                    'shape': wshape
                })
            else:
                num = layer.split('_')[-1]
                if mod == 'depthwise':
                    kind = 'DepthwiseConv2D'
                elif mod == 'expand' or mod == 'project':
                    kind = 'Conv2D'

                    # then layer is depthwise
                w_layer_guide.append({
                    "key": wkey,
                    'block_id': num,
                    'layer': kind,
                    'mod': mod,
                    'meta': meta,
                    'shape': wshape
                })

        # Merge layer guides together
        layer_guide = bn_layer_guide + w_layer_guide + not_expanded_layers

        # Sort the layer guide by block_ids
        layer_guide = sorted(layer_guide, key=lambda x: int(x['block_id']))


        # Save layer guide to np arrays
        for layer in layer_guide:
            T = reader.get_tensor(layer['key'])

            filename = layer['key'].replace('/', '_')
            filename = filename.replace(
                'MobilenetV2_', './weights'+str(alpha)+str(rows)+'/') + '.npy'
            if not os.path.isdir('./weights' + str(alpha)+str(rows)):
                os.makedirs('./weights' + str(alpha)+str(rows))
            layer['file'] = filename
            np.save(filename, T)

        print('len of layer_guide: ', len(layer_guide))


        # Save to dir
        extraction_repo_path = '/home/jon/Documents/keras_mobilenetV2/'
        with open(extraction_repo_path + 'layer_guide' + str(alpha) + str(rows) + '.p', 'wb') as pickle_file:
            pickle.dump(layer_guide, pickle_file)
        
        print('created for : ', alpha, 'rows: ', rows)



if __name__ == "__main__":
    # models = [(1.0, 128)]
    extract_weights(models = models_to_load)
    print('weights extracted')
