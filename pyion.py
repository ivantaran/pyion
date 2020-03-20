
from matplotlib import pyplot
from pyion_utils import *
from pyion_tf import *
import os
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data = pyion_load('/home/taran/work/pyion/delta1.txt')
m_train = data['azm'].shape[0] - 100
m_test = 100

X_train = np.stack((data['azm'][0:m_train], data['elv'][0:m_train]))
Y_train = np.reshape(data['ion'][0:m_train], (1, m_train))

X_test = np.stack((data['azm'][m_train:m_train + m_test], data['elv'][m_train:m_train + m_test]))
Y_test = np.reshape(data['ion'][m_train:m_train + m_test], (1, m_test))

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150, minibatch_size=32)

m = 1000
azm = np.zeros((1, m), dtype=np.float32) + 2.7
elv = np.linspace(0.0, np.pi * 0.5, m, dtype=np.float32).reshape((1, m))
mx = np.concatenate((azm, elv))
print(mx.shape)

ion = forward_propagation(X_train, parameters)
pyplot.plot(np.squeeze(ion.numpy()), '.')
pyplot.plot(np.squeeze(Y_train), '.')
pyplot.show()
# pyion_plotmap(data, 1024)
exit(0)

