from __future__ import print_function

import numpy as np
import pickle

from keras.layers import Input, Conv2D, Dense
from mobilenetv2 import MobileNetV2, BatchNorm, DepthwiseConv2D

alpha = 1.3
rows = 224


WEIGHTS_SAVE_PATH_INCLUDE_TOP = '/home/jon/Documents/keras_mobilenetV2/test_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(rows) + '.h5'
WEIGHTS_SAVE_PATH_NO_TOP = '/home/jon/Documents/keras_mobilenetV2/test_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(rows) + '_no_top' + '.h5'


with open('layer_guide'+str(alpha) + str(rows) +  '.p', 'rb') as pickle_file:
	layer_guide = pickle.load(pickle_file)

# layers 0-4 are first conv
# layers 4 - 16 are expanded

# load mobilenet
input_tensor = Input(shape = (rows, rows, 3))
model = MobileNetV2(input_tensor = input_tensor, include_top=True, alpha = alpha, weights = None)


not_expanded_layers = ['bn_Conv1',
                       'Conv_1_bn',
                       'Conv1',
                       'Logits',
                       'bn_Conv1',
                       'Logits',
                       'Conv_1',
                       'Conv_1_bn',
                       'Conv_1_bn',
                       'bn_Conv1',
                       'bn_Conv1',
                       'Conv_1_bn']


def find_not_exp_conv_layer_match(keras_layer, kind=None):
	"""
	This function takes in layer parameters and finds the associated conv layer from the weights layer guide
	Returns the weights layer guide object , found boolean
	"""
	for lobj in layer_guide:
		if keras_layer.name == lobj['layer']:
			if keras_layer.weights[0]._keras_shape == lobj['shape']:
				return lobj, True
			else:
				return {}, False


def find_not_exp_bn_layers_match(keras_layer, kind=None):
	"""
	This function takes in layer parameters and finds the associated conv layer from the weights layer guide
	Returns the weights layer guide object , found boolean
	"""
	meta_full = 4
	metas_found = 0

	for lobj in layer_guide:
		if keras_layer.name == lobj['layer']:
			if lobj['meta'] == 'gamma' and keras_layer.weights[0]._keras_shape == lobj['shape']:
				gamma = lobj
				metas_found += 1
			elif lobj['meta'] == 'beta' and keras_layer.weights[1]._keras_shape == lobj['shape']:
				beta = lobj
				metas_found += 1
			elif lobj['meta'] == 'moving_mean' and keras_layer.weights[2]._keras_shape == lobj['shape']:
				moving_mean = lobj
				metas_found += 1
			elif lobj['meta'] == 'moving_variance' and keras_layer.weights[3]._keras_shape == lobj['shape']:
				moving_variance = lobj
				metas_found += 1

	if metas_found == meta_full:
		return [gamma, beta, moving_mean, moving_variance], True
	else:
		return [], False


def find_not_exp_dense_layers_match(keras_layer, kind=None):
	meta_full = 2
	metas_found = 0
	for lobj in layer_guide:
		if keras_layer.name == lobj['layer']:
			if lobj['meta'] == 'weights':
				weights = lobj
				metas_found += 1
			elif lobj['meta'] == 'biases':
				bias = lobj
				metas_found += 1
	if metas_found == meta_full:
		return [weights, bias], True
	else:
		return [], False



def find_conv_layer_match(keras_layer, block_id = None, mod = None, kind = None):
	"""
	This function takes in layer parameters and finds the associated conv layer from the weights layer guide
	Returns the weights layer guide object , found boolean
	"""
	for lobj in layer_guide:
		if int(lobj['block_id']) == int(block_id):
			if lobj['layer'] == kind:
				if lobj['mod'] == mod:
					if keras_layer.weights[0]._keras_shape == lobj['shape']:
						return lobj, True
					else:
						return {}, False

def find_bn_layers_match(keras_layer, block_id = None, mod = None, kind = None):
	"""
	This function takes in layer parameters and returns a list of the four batch norm parameters as well as a boolean indicating success
	"""
	meta_full = 4
	metas_found = 0
	
	for lobj in layer_guide:
		if int(lobj['block_id']) == int(block_id):
			if lobj['layer'] == kind:
				if lobj['mod'] == mod:
					if lobj['meta'] == 'gamma' and keras_layer.weights[0]._keras_shape == lobj['shape']:
						gamma = lobj
						metas_found += 1
					elif lobj['meta'] == 'beta' and keras_layer.weights[1]._keras_shape == lobj['shape']:
						beta = lobj
						metas_found += 1
					elif lobj['meta'] == 'moving_mean' and keras_layer.weights[2]._keras_shape == lobj['shape']:
						moving_mean = lobj
						metas_found += 1
					elif lobj['meta'] == 'moving_variance' and keras_layer.weights[3]._keras_shape == lobj['shape']:
						moving_variance = lobj 
						metas_found += 1
	if metas_found == meta_full:
		return [gamma, beta, moving_mean, moving_variance], True
	else:
		return [], False

# Calculate loaded number
set_weights = 0
for keras_layer in model.layers:
	name = keras_layer.name.split('_')
	# If it not an expandable layer
	if keras_layer.name in not_expanded_layers:
		print('keras_layer : ', keras_layer.name, ' Is not expandable')
		if isinstance(keras_layer, BatchNorm):
			bn_layers, isfound = find_not_exp_bn_layers_match(keras_layer = keras_layer, kind='BatchNorm')
			if isfound:
				arrs = [np.load(lobj['file']) for lobj in bn_layers]
				keras_layer.set_weights(arrs)
				set_weights += 1 # can add ()
			else:
				print('possible error not match found if isinstance BatchNorm and not expandable')

		elif isinstance(keras_layer, DepthwiseConv2D):
			lobj, isfound = find_not_exp_conv_layer_match(
				keras_layer=keras_layer, kind='DepthwiseConv2D')
			if isfound:
				if lobj['meta'] == 'weights':
					arr = np.load(lobj['file'])
					keras_layer.set_weights([arr])
					set_weights += 1
			else:
				print(' You probably wont see this but just in case possible error finding weights for not expandable DepthwiseConv2D: ')
		elif isinstance(keras_layer, Conv2D):
			# get mods

			lobj, isfound = find_not_exp_conv_layer_match(keras_layer=keras_layer, kind='Conv2D')
			if isfound:
				if lobj['meta'] == 'weights':
					arr = np.load(lobj['file'])
					keras_layer.set_weights([arr])
					set_weights += 1
			else:
				print('possible error finding weights for not expandable Conv2D: ')
				# if layer['meta'] == 'biases'
		elif isinstance(keras_layer, Dense):
			dense_layers, isfound = find_not_exp_dense_layers_match(keras_layer=keras_layer, kind='Dense')
			if isfound:
				weights = np.load(dense_layers[0]['file'])
				bias = np.load(dense_layers[1]['file'])
				# Remove background classes

				# squeeze
				weights = np.squeeze(weights)


				keras_layer.set_weights([weights, bias])
				set_weights += 1
			else:
				print('possible error with dense layer in not expandable')

	else:
		if isinstance(keras_layer, BatchNorm):
			_, _, num, _, mod = name
			bn_layers, isfound = find_bn_layers_match(keras_layer=keras_layer, block_id=num, mod=mod, kind='BatchNorm')
			# note: BatchNorm layers have 4 weights
			# This will return a list of the 4 objects corresponding to the 4 weights in the right order
			# Then go through the list and grab each numpy array, make sure it is wrapped inside a standard list
			if isfound:
				arrs = [np.load(lobj['file']) for lobj in bn_layers]
				keras_layer.set_weights(arrs)
				set_weights += 1
			else:
				print('possible error not match found on expandable batchNorm layers')
				print('=== dump: ')
				print('keras_layer: ', keras_layer.name)
				
	
		# Check for DepthwiseConv2D
		elif isinstance(keras_layer, DepthwiseConv2D):
			_, _, num, mod = name
			# check layer meta to be depthwise_weights
			layer, isfound = find_conv_layer_match(keras_layer=keras_layer, block_id=num, mod=mod, kind='DepthwiseConv2D')
			if isfound:
				if layer['meta'] == 'depthwise_weights':
					arr = np.load(layer['file'])
					keras_layer.set_weights([arr])
					set_weights += 1
			else:
				print('possible error not match found for DepthwiseConv2D in not expandable')
			
		# Check for Conv2D
		elif isinstance(keras_layer, Conv2D):
			_, _, num, mod = name
			layer, isfound = find_conv_layer_match(keras_layer=keras_layer, block_id=num, mod=mod, kind='Conv2D')
			if isfound:
				if layer['meta'] == 'weights':
					arr = np.load(layer['file'])
					keras_layer.set_weights([arr])
			else:
				print('possible error not match found')

# Set first block

# Organize batch norm layers into [gamma, beta, moving_mean, moving_std]
# Organize expand conv block layers into [expand, depthwise, project]
print('set_weights: ', set_weights)
model.save_weights(WEIGHTS_SAVE_PATH_INCLUDE_TOP)

# Save no top model
out_relu = model.get_layer('out_relu').output
from keras.models import Model
model_no_top = Model(input = input_tensor, output = out_relu)
model_no_top.save(WEIGHTS_SAVE_PATH_NO_TOP)


trainable_layers = [l.weights for l in model.layers]
print('len trainable: ', len(trainable_layers))

# TODO test predict with detect api
print('model: ', model)
print('set_weights: ', set_weights)
assert(set_weights == 72)
print('alpha: ', alpha)
print('rows: ', rows)



