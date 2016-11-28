# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import cv2
from keras.applications.xception import Xception

# some preprocessing
image = cv2.imread('../1.jpg')
image = cv2.resize(image, (299, 299))
image.shape = (1,) + image.shape
image = image.astype(np.float32)
image /= 255.

# keras prediction
model = Xception(include_top=True, weights='imagenet', input_tensor=None)
out = model.predict(image)
print('keras:', out.argmax(), out[0, out.argmax()])


# mxnet prediction
symbol, arg_params, aux_params = mx.model.load_checkpoint('xception', 1)
mod = mx.module.Module(symbol)
mod.bind(data_shapes=[('data', (1, 3, 299, 299))], for_training=False)
mod.init_params(arg_params=arg_params, aux_params=aux_params)

x = mx.io.NDArrayIter(np.transpose(image, (0, 3, 1, 2)))
out = mod.predict(x).asnumpy()

print('mxnet:', out.argmax(), out[0, out.argmax()])
