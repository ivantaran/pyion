
from matplotlib import pyplot
from pyion_utils import *
from pyion_tf import *
import os
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


n = 1000
m = 1000
# mx, my = pyion_debug_samples(n, azm_range=(-0.25, 0.25))
# mx, my = pyion_debug_samples_m(m, azm_range=(-0.25, 0.25))
# mx, my = pyion_debug_samples2(n)
# mx, my = pyion_debug_samples2_m(m)

# mx, my = pyion_shuffle_samples(mx, my)
# m_train = int(m * 0.8)
# m_test = m - m_train

# mx_train = mx[:, :m_train]
# my_train = my[:, :m_train]
# mx_test = mx[:, m_train:]
# my_test = my[:, m_train:]
# print('train set shapes: ', mx_train.shape, my_train.shape)
# print('test set shapes: ', mx_test.shape, my_test.shape)

# pyplot.plot(np.squeeze(mx[1]), np.squeeze(my), '.')
# pyplot.show()
# exit(0)

mx, my = pyion_load('/home/taran/work/pyion/delta.txt')

# m_train = 1000 #data['azm'].shape[0] // 10
# m_test = 1000
#
# X_train = np.stack((data['azm'][0:m_train], data['elv'][0:m_train]))
# Y_train = np.reshape(data['ion'][0:m_train], (1, m_train))
#
# X_test = np.stack((data['azm'][m_train:m_train + m_test], data['elv'][m_train:m_train + m_test]))
# Y_test = np.reshape(data['ion'][m_train:m_train + m_test], (1, m_test))
#
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

preload_weights = True
parameters = model(mx, my, mx, my, learning_rate=0.00001, num_epochs=100, minibatch_size=my.shape[1] // 2,
                   preload_weights=preload_weights)

np.save('parameters1.npy', parameters)

# m = 1000
# azm = np.zeros((1, m), dtype=np.float32) + 2.7
# elv = np.linspace(0.0, np.pi * 0.5, m, dtype=np.float32).reshape((1, m))
# mx = np.concatenate((azm, elv))
# print(mx.shape)

ion = forward_propagation(mx, parameters)
t = np.reshape(mx, (2, -1))
pyplot.plot(mx[1, :], np.squeeze(ion.numpy()), '.')
pyplot.plot(mx[1, :], np.ravel(my) + 0.1, '.')
# pyplot.plot(t[0, :], '.')
# pyplot.plot(t[1, :], '.')
pyplot.gcf().set_size_inches(16, 8)
pyplot.grid(True)
pyplot.show()


# pyion_plotmap(data, 1024)
exit(0)

