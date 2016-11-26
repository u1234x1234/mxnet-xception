import cv2
import numpy as np
import joblib
from keras.applications.xception import Xception
from keras.utils.visualize_util import plot
from keras.utils import generic_utils
from keras.models import Model
from keras import backend as K


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations


image = cv2.imread('1.jpg')
image = cv2.resize(image, (299, 299))

#x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
x = image.astype(np.float32)
#x = x.transpose((2, 0, 1))
x = x.reshape((1,) + x.shape)
x /= 255.

model = Xception(include_top=True, weights='imagenet', input_tensor=None)

weights = dict()
for layer in model.layers:
    for e in zip(layer.weights, layer.get_weights()):
#        print('{} : {}'.format(e[0].name, e[1].shape))
        weights[e[0].name] = e[1]
#joblib.dump(weights, 'd2.pkl')

#import tensorflow as tf
#tf.ops.variables.Variable.
#qwe

layer2id = {l.name: i for i, l in enumerate(model.layers)}

#d = {}
#for layer in model.layers:
#    layer_name = layer.name
#    layer_weights = layer.get_weights()
#    d[layer_name] = layer_weights
#
#joblib.dump(d, 'd.pkl')
    
    
out = model.predict(x)
print(out.argmax())

act = get_activations(model, layer2id['block14_sepconv2_act'], x)
assert len(act) == 1
r = act[0]
print(r.shape)
print(np.sum(r[0,:,:,0]))
print(np.sum(r[0,:,:,37]))