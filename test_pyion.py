
import unittest
import pyion_tf
import numpy as np
import tensorflow as tf

class TestPyoin(unittest.TestCase):

    def test_cost(self):
        Y = np.array([0.1, 0.2, 0.3])
        Y_pred = np.array([1.0, 2.0, 3.0])
        test = np.sum(np.square(Y - Y_pred)) / Y.shape[0]

        tf.reset_default_graph()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, name='x')
            y = tf.placeholder(tf.float32, name='y')
            cost = pyion_tf.compute_cost(x, y)
            cost = sess.run(cost, feed_dict={x: Y_pred, y: Y})
            sess.close()
            self.assertAlmostEqual(cost, test, delta=1.0e-7)

    def test_create_placeholders(self):
        tf.reset_default_graph()
        X, Y = pyion_tf.create_placeholders(2, 1)
        self.assertEqual(str(X), 'Tensor("X:0", shape=(2, ?), dtype=float32)')
        self.assertEqual(str(Y), 'Tensor("Y:0", shape=(1, ?), dtype=float32)')
    
    def test_initialize_parameters(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            parameters = pyion_tf.initialize_parameters()
            sess.close()
            self.assertEqual(str(parameters['W1']), "<tf.Variable 'W1:0' shape=(4, 2) dtype=float32_ref>")
    
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
