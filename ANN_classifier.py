import tensorflow as tf
import numpy as np


D = 2 # dimensionality of input
M = 3 # hidden layer size
K = 3 # number of classes

def init_weights(shape, **kwargs):

    name = ""
    var_name = kwargs['name']
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name = var_name)


def feedforward(X, W1, b1, W2, b2):
    Z = tf.nn.relu(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2

class Text_ANN(object):

    def __init__(self, num_features, num_classes, hidden_layer_size,  l2_reg_lambda=0.0):


        self.input_x = tf.placeholder(tf.float32, [None,num_features])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32, "dropout_keep_prob")

        W1 = init_weights([num_features, hidden_layer_size], name = "W1") # create symbolic variables
        b1 = init_weights([ hidden_layer_size])
        W2 = init_weights([ hidden_layer_size, num_classes], name= "W2")
        b2 = init_weights([num_classes])

        # Getting full layer weights

        weights_full_layer = tf.get_variable(
                                    "W2",
                                    shape = [hidden_layer_size, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer()
                                )

        logits = forward(self.input_x, W1, b1, W2, b2)

        with tf.name_scope("loss"):
            l2_reg_weights = tf.nn.l2_loss(weights_full_layer)
            cross_enthropy_loss =  tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.input_y,
                logits=logits
              )

              self.loss = tf.reduce_mean(cross_enthropy_loss + (l2_reg_lambda * l2_reg_weights))


        # compute cross_enthropy_losss
        # WARNING: This op expects unscaled logits,
        # since it performs a softmax on logits
        # internally for efficiency.
        # Do not call this op with the output of softmax,
        # as it will produce incorrect results.

        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cross_enthropy_loss) # construct an optimizer
        # input parameter is the learning rate

        predict_op = tf.argmax(logits, 1)
        # input parameter is the axis on which to choose the max

        # just stuff that has to be done
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
