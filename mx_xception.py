# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import joblib
import cv2
import re

weights = joblib.load('d2.pkl')

keras_arg_params = dict()
keras_aux_params = dict()

for l in weights:
    w = weights[l]
    if len(w.shape) == 4:  # (in_c,out_c,y,x) -> (out_c,in_c,y,x)
        w = np.transpose(w, (3, 2, 0, 1))
    if len(w.shape) == 2:
        w = np.transpose(w, (1, 0))
#        w = w.reshape((1000, ))

    layer_name = l[:-2]  # remove ':0' at the end of the layer name
    layer_name = layer_name.replace('_W', '_weight')
    layer_name = re.sub('_b$', '_bias', layer_name)

    layer_name = layer_name.replace('_running_mean', '_moving_mean')
    layer_name = layer_name.replace('_running_std', '_moving_var')

    layer_name = layer_name.replace('_pointwise_kernel',
                                    '_pointwise_kernel_weight')

    if layer_name.endswith('_depthwise_kernel'):
        n_channel = w.shape[1]
        w = np.split(w, n_channel, axis=1)
        for i in range(n_channel):
            keras_arg_params[layer_name+str(i)+'_weight'] = mx.nd.array(w[i])
        continue

    w = mx.nd.array(w)

    if layer_name.endswith('_mean') or layer_name.endswith('_var'):
        keras_aux_params[layer_name] = w
    else:
        keras_arg_params[layer_name] = w

    print(layer_name, w.shape)


image = cv2.imread('1.jpg')
image = cv2.resize(image, (299, 299))
x = np.transpose(image, (2, 0, 1))
x.shape = (1,) + x.shape
x = x.astype(np.float32)
x /= 255.
print(x.shape)


def separable_conv(data, num_in_ch, num_out_ch, kernel, pad, name, depth_mult=1):
    #  depthwise convolution
    channels = mx.sym.SliceChannel(data, axis=1, num_outputs=num_in_ch)
    dw_outs = [mx.sym.Convolution(data=channels[i], num_filter=depth_mult,
                                  kernel=(3, 3), pad=pad, no_bias=True,
                                  name=name+'_depthwise_kernel'+str(i))
               for i in range(num_in_ch)]
    dw_out = mx.sym.Concat(*dw_outs)
    #  pointwise convolution
    pw_out = mx.sym.Convolution(dw_out, num_filter=num_out_ch, kernel=(1, 1),
                                no_bias=True, name=name+'_pointwise_kernel')
    return pw_out

data = mx.sym.Variable('data')
bl1 = mx.sym.Convolution(data, num_filter=32, kernel=(3, 3), stride=(2, 2),
                         no_bias=True, name='block1_conv1')
bl1 = mx.sym.BatchNorm(bl1, name='block1_conv1_bn')
bl1 = mx.sym.Activation(bl1, act_type='relu', name='block1_conv1_act')
bl1 = mx.sym.Convolution(bl1, num_filter=64, kernel=(3, 3), stride=(1, 1),
                         no_bias=True, name='block1_conv2')
bl1 = mx.sym.BatchNorm(bl1, name='block1_conv2_bn')
bl1 = mx.sym.Activation(bl1, act_type='relu', name='block1_conv2_act')

# block 2

res2 = mx.sym.Convolution(bl1, num_filter=128, kernel=(1, 1), stride=(2, 2),
                          no_bias=True, name='convolution2d_1')
res2 = mx.sym.BatchNorm(res2, name='batchnormalization_1')

bl2 = separable_conv(bl1, 64, 128, (3, 3), (1, 1), 'block2_sepconv1')
bl2 = mx.sym.BatchNorm(bl2, name='block2_sepconv1_bn')
bl2 = mx.sym.Activation(bl2, act_type='relu', name='block2_sepconv1_act')
bl2 = separable_conv(bl2, 128, 128, (3, 3), (1, 1), 'block2_sepconv2')
bl2 = mx.sym.BatchNorm(bl2, name='block2_sepconv2_bn')

bl2 = mx.sym.Pooling(bl2, kernel=(3, 3), stride=(2, 2), pool_type='max',
                     pad=(1, 1), name='block2_pool')
bl2 = bl2 + res2

# block 3

res3 = mx.sym.Convolution(bl2, num_filter=256, kernel=(1, 1), stride=(2, 2),
                          no_bias=True, name='convolution2d_2')
res3 = mx.sym.BatchNorm(res3, name='batchnormalization_2')

bl3 = separable_conv(bl2, 128, 256, (3, 3), (1, 1), 'block3_sepconv1')
bl3 = mx.sym.BatchNorm(bl3, name='block3_sepconv1_bn')
bl3 = mx.sym.Activation(bl3, act_type='relu', name='block3_sepconv1_act')
bl3 = separable_conv(bl3, 256, 256, (3, 3), (1, 1), 'block3_sepconv2')
bl3 = mx.sym.BatchNorm(bl3, name='block3_sepconv2_bn')

bl3 = mx.sym.Pooling(bl3, kernel=(3, 3), stride=(2, 2), pool_type='max',
                     pad=(1, 1), name='block3_pool')
bl3 = bl3 + res3

# block 4

res4 = mx.sym.Convolution(bl3, num_filter=728, kernel=(1, 1), stride=(2, 2),
                          no_bias=True, name='convolution2d_3')
res4 = mx.sym.BatchNorm(res4, name='batchnormalization_3')

bl4 = separable_conv(bl3, 256, 728, (3, 3), (1, 1), 'block4_sepconv1')
bl4 = mx.sym.BatchNorm(bl4, name='block4_sepconv1_bn')
bl4 = mx.sym.Activation(bl4, act_type='relu', name='block4_sepconv1_act')
bl4 = separable_conv(bl4, 728, 728, (3, 3), (1, 1), 'block4_sepconv2')
bl4 = mx.sym.BatchNorm(bl4, name='block4_sepconv2_bn')

bl4 = mx.sym.Pooling(bl4, kernel=(3, 3), stride=(2, 2), pool_type='max',
                     pad=(1, 1), name='block4_pool')
bl4 = bl4 + res4

for i in range(8):
    residual = bl4
    prefix = 'block' + str(i + 5)

    bl = mx.sym.Activation(bl4, act_type='relu', name=prefix+'_sepconv1_act')
    bl = separable_conv(bl, 728, 728, (3, 3), (1, 1), prefix+'_sepconv1')
    bl = mx.sym.BatchNorm(bl, name=prefix+'_sepconv1_bn')
    bl = mx.sym.Activation(bl, act_type='relu', name=prefix+'_sepconv2_act')
    bl = separable_conv(bl, 728, 728, (3, 3), (1, 1), prefix+'_sepconv2')
    bl = mx.sym.BatchNorm(bl, name=prefix+'_sepconv2_bn')
    bl = mx.sym.Activation(bl, act_type='relu', name=prefix+'_sepconv3_act')
    bl = separable_conv(bl, 728, 728, (3, 3), (1, 1), prefix+'_sepconv3')
    bl = mx.sym.BatchNorm(bl, name=prefix+'_sepconv3_bn')

    bl = bl + residual

res5 = mx.sym.Convolution(bl, num_filter=1024, kernel=(1, 1), stride=(2, 2),
                          no_bias=True, name='convolution2d_4')
res5 = mx.sym.BatchNorm(res5, name='batchnormalization_4')

bl13 = mx.sym.Activation(bl, act_type='relu', name='block13_sepconv1_act')
bl13 = separable_conv(bl13, 728, 728, (3, 3), (1, 1), 'block13_sepconv1')
bl13 = mx.sym.BatchNorm(bl13, name='block13_sepconv1_bn')
bl13 = mx.sym.Activation(bl13, act_type='relu', name='block13_sepconv2_act')
bl13 = separable_conv(bl13, 728, 1024, (3, 3), (1, 1), 'block13_sepconv2')
bl13 = mx.sym.BatchNorm(bl13, name='block13_sepconv2_bn')

bl13 = mx.sym.Pooling(bl13, kernel=(3, 3), stride=(2, 2), pool_type='max',
                      pad=(1, 1), name='block13_pool')
bl13 = bl13 + res5

bl14 = separable_conv(bl13, 1024, 1536, (3, 3), (1, 1), 'block14_sepconv1')
bl14 = mx.sym.BatchNorm(bl14, name='block14_sepconv1_bn')
bl14 = mx.sym.Activation(bl14, act_type='relu', name='block14_sepconv1_act')
bl14 = separable_conv(bl14, 1536, 2048, (3, 3), (1, 1), 'block14_sepconv2')
bl14 = mx.sym.BatchNorm(bl14, name='block14_sepconv2_bn')
bl14 = mx.sym.Activation(bl14, act_type='relu', name='block14_sepconv2_act')

pool = mx.sym.Pooling(bl14, kernel=(10, 10), global_pool=True, pool_type='avg',
                      name='global_pool')
fc = mx.symbol.FullyConnected(data=pool, num_hidden=1000, name='predictions')
softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')

mod = mx.module.Module(softmax)
mod.bind(data_shapes=[('data', (1, 3, 299, 299))], for_training=False)
mod.init_params(arg_params=keras_arg_params, aux_params=keras_aux_params)


print('mxnet init args:')
args, auxs = mod.get_params()
for l in args:
    print(l, args[l].shape)
for l in auxs:
    print(l, auxs[l].shape)


x = mx.io.NDArrayIter(x)
out = mod.predict(x).asnumpy()

print(out.shape)
print(np.sum(out[0, 0]))
print(np.sum(out[0, 37]))
print(out.argmax())
