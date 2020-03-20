
from pyion_utils import *
from pyion_tf import *
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data = pyion_load('/home/taran/work/pyion/delta1.txt')
m_train = data['azm'].shape[0] - 1000
m_test = 100

X_train = np.stack((data['azm'][0:m_train], data['elv'][0:m_train]))
Y_train = np.reshape(data['ion'][0:m_train], (1, m_train))

X_test = np.stack((data['azm'][m_train:m_train + m_test], data['elv'][m_train:m_train + m_test]))
Y_test = np.reshape(data['ion'][m_train:m_train + m_test], (1, m_test))

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=1500, minibatch_size=64)

# pyion_plotmap(data, 1024)
exit(0)

