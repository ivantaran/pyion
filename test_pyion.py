
import unittest
import pyion_tf
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestPyoin(unittest.TestCase):

    def test_cost(self):
        Y = np.array([0.1, 0.2, 0.3])
        Y_pred = np.array([1.0, 2.0, 3.0])
        test = np.sum(np.square(Y - Y_pred)) / Y.shape[0]

        cost = pyion_tf.compute_cost(Y_pred, Y)
        self.assertAlmostEqual(cost.numpy(), test, delta=1.0e-7)

    # def test_create_placeholders(self):
    #     tf.reset_default_graph()
    #     X, Y = pyion_tf.create_placeholders(2, 1)
    #     self.assertEqual(str(X), 'Tensor("X:0", shape=(2, ?), dtype=float32)')
    #     self.assertEqual(str(Y), 'Tensor("Y:0", shape=(1, ?), dtype=float32)')

    def test_initialize_parameters(self):
        parameters = pyion_tf.initialize_parameters()
        self.assertEqual(str(parameters.keys()), "dict_keys(['W1', 'b1', 'W2', 'b2', 'W3', 'b3'])")
    
    def test_forward_propagation(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            X, Y = pyion_tf.create_placeholders(2, 1)
            parameters = pyion_tf.initialize_parameters()
            Y_pred = pyion_tf.forward_propagation(X, parameters)
            sess.close()
            self.assertEqual(str(Y_pred), 'Tensor("add_2:0", shape=(1, ?), dtype=float32)')

    
if __name__ == '__main__':
    unittest.main()
