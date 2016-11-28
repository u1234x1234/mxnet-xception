# -*- coding: utf-8 -*-
"""
Model Xception conversion from keras to mxnet
keras.io/
mxnet.io/
"""
import re
import mxnet as mx
import numpy as np
from keras.applications.xception import Xception
from symbol_xception import get_xception_symbol


# load Xception model
model = Xception(include_top=True, weights='imagenet', input_tensor=None)

weights = dict()
# dump all weights (trainable and not) to dict {layer_name: layer_weights}
for layer in model.layers:
    for layer, layer_weights in zip(layer.weights, layer.get_weights()):
        weights[layer.name] = layer_weights


# we need two dicts: one for trainable params (arg_params)
# and one for BN layer statistics: _mean, _var
keras_arg_params = dict()
keras_aux_params = dict()

for l in weights:
    w = weights[l]

    # BHWC layout to BCHW
    if len(w.shape) == 4:  # (in_c,out_c,y,x) -> (out_c,in_c,y,x)
        w = np.transpose(w, (3, 2, 0, 1))
    if len(w.shape) == 2:  # transpose fc layer
        w = np.transpose(w, (1, 0))

    layer_name = l[:-2]  # remove ':0' at the end of the layer name

    layer_name = re.sub('_W$', '_weight', layer_name)
    layer_name = re.sub('_b$', '_bias', layer_name)

    layer_name = re.sub('_running_mean$', '_moving_mean', layer_name)
    layer_name = re.sub('_running_std$', '_moving_var', layer_name)

    layer_name = re.sub('_pointwise_kernel$', '_pointwise_kernel_weight',
                        layer_name)

    #  There's no depthwise conv layer in mxnet, so we just split one kernel
    #  into n 1-dimensional convolutions
    if layer_name.endswith('_depthwise_kernel'):
        n_channel = w.shape[1]
        w = np.split(w, n_channel, axis=1)
        for i in range(n_channel):
            keras_arg_params[layer_name+str(i)+'_weight'] = mx.nd.array(w[i])
        continue

    if layer_name.endswith('_mean') or layer_name.endswith('_var'):  # BN layer
        keras_aux_params[layer_name] = mx.nd.array(w)
    else:
        keras_arg_params[layer_name] = mx.nd.array(w)

mx.model.save_checkpoint('xception', 1, get_xception_symbol(),
                         keras_arg_params, keras_aux_params)
