
from pyion_utils import *
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.initializers import glorot_uniform

# tf.compat.v1.enable_eager_execution()


@tf.function
def compute_cost(Y_pred, Y):
    cost = tf.reduce_mean(tf.math.squared_difference(Y, Y_pred))
    return cost


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    return X, Y


def initialize_parameters():
    n_x = 2
    n1 = 4
    n2 = 4
    n_y = 1
    # tf.set_random_seed(1)
    W1 = tf.Variable(glorot_uniform()((n1, n_x)), name='W1')
    b1 = tf.Variable(glorot_uniform()((n1, 1)), name='b1', trainable=True)
    W2 = tf.Variable(glorot_uniform()((n2, n1)), name='W2', trainable=True)
    b2 = tf.Variable(glorot_uniform()((n2, 1)), name='b2', trainable=True)
    W3 = tf.Variable(glorot_uniform()((n_y, n2)), name='W3', trainable=True)
    b3 = tf.Variable(glorot_uniform()((n_y, 1)), name='b3', trainable=True)

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2,
                  'W3': W3,
                  'b3': b3}

    return parameters

@tf.function
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    return Z3


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = m // mini_batch_size # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# @tf.function
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True):
    
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    # X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    # Z3 = forward_propagation(X, parameters)
    # cost = compute_cost(Z3, Y)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#.minimize(cost)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)#.minimize(cost)

    for epoch in range(num_epochs):
        epoch_cost = 0.0
        num_minibatches = int(m / minibatch_size)
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            Z3 = forward_propagation(minibatch_X, parameters)
            minibatch_cost = compute_cost(Z3, minibatch_Y)
            minimize_op = optimizer.minimize(compute_cost, var_list=tf.trainable_variables)
            epoch_cost += minibatch_cost / minibatch_size

        if print_cost == True and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    parameters = sess.run(parameters)
    print ("Parameters have been trained!")

    # correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy = compute_cost(Z3, Y)
    # print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters
