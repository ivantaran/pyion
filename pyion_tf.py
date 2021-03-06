
from pyion_utils import *
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.initializers import glorot_uniform

tf.compat.v1.enable_eager_execution()


@tf.function
def compute_cost(Y_pred, Y):
    cost = tf.reduce_mean(tf.math.squared_difference(Y, Y_pred))
    return cost


def initialize_parameters(nl=[2, 100, 1], preload_weights=False):
    if preload_weights:
        parameters = np.load('parameters.npy', allow_pickle=True)
        parameters = parameters.tolist()
    else:
        parameters = {}

        for l in range(1, len(nl)):
            n_prev = nl[l - 1]
            n = nl[l]
            parameters['W%d' % l] = tf.Variable(glorot_uniform()((n, n_prev)))
            parameters['b%d' % l] = tf.Variable(glorot_uniform()((n, 1)))

    return parameters


@tf.function
def forward_propagation(mx, parameters):
    nl = len(parameters) // 2
    A = mx

    for l in range(1, nl):
        W = parameters['W%d' % l]
        b = parameters['b%d' % l]
        Z = tf.add(tf.matmul(W, A), b)
        A = tf.nn.relu(Z) if l == 1 else tf.nn.relu(Z)

    W = parameters['W%d' % nl]
    b = parameters['b%d' % nl]

    Z = tf.add(tf.matmul(W, A), b)

    return Z


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
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
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# @tf.function
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.000003, num_epochs=1500, minibatch_size=32, print_cost=True,
          preload_weights=False):
    
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    parameters = initialize_parameters([n_x, 1024, 256, 256, 256, 256, n_y], preload_weights=preload_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        epoch_cost = 0.0
        num_minibatches = m // minibatch_size
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            with tf.GradientTape() as tape:
                (minibatch_X, minibatch_Y) = minibatch
                Z3 = forward_propagation(minibatch_X, parameters)
                minibatch_cost = compute_cost(Z3, minibatch_Y)
            gradients = tape.gradient(minibatch_cost, list(parameters.values()))
            optimizer.apply_gradients(zip(gradients, list(parameters.values()))) #tf.trainable_variables()
            epoch_cost += minibatch_cost

        if print_cost is True and epoch % 1 == 0:
            print("Cost after epoch %i: %0.3e" % (epoch, epoch_cost))
            np.save('parameters1.npy', parameters)
        if print_cost is True and epoch % 5 == 0:
            costs.append(epoch_cost.numpy())

    plt.plot(10.0 * np.log10(np.squeeze(costs)))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    print("Parameters have been trained!")

    # correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    Z3 = forward_propagation(X_train, parameters)
    accuracy = compute_cost(Z3, Y_train)
    print ("Train Accuracy:", accuracy.numpy())
    # print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters
