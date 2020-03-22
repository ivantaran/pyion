
from matplotlib import pyplot
from pyion_utils import *
from pyion_tf import *
import os
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
k = 8
mx = pyion_azmelv_mesh(360 * k, 90 * k)

parameters = np.load('parameters.npy', allow_pickle=True)
parameters = parameters.tolist()

ion = forward_propagation(mx, parameters)

data = {
    'azm': mx[0, :],
    'elv': mx[1, :],
    'ion': ion.numpy()[0, :],
}

pyion_plotmap(data, 512, cmap='terrain')

# # pyplot.plot(parameters['W4'].numpy().T)
# pyplot.plot(sorted(parameters['W1'][:, 1].numpy()))
# pyplot.imshow(parameters['W1'].numpy())
# pyplot.plot(parameters['W1'][:, 1].numpy(), parameters['b1'].numpy(), '.')
# pyplot.plot(parameters['W1'].numpy())
# pyplot.plot(parameters['W1'][:, 0].numpy(), parameters['W1'][:, 1].numpy(), '.')
# pyplot.show()
# exit(0)
