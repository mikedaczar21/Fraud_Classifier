from joblib import dump, load
import tensorflow as tf
import numpy as np
import dill

# Helper Functions to create convolution neural network
def init_weights(input_shape):
    init_random_dist = tf.truncated_normal(input_shape, stddev=0.1, name="weights")
    return tf.Variable(init_random_dist)

def init_bias(input_shape):
    init_bias = tf.constant(0.1, shape= [input_shape], name="bias")
    return init_bias

def conv2d(input_tensor, input_filter, input_strides, input_padding):
    return tf.nn.conv2d(input= input_tensor, filter=input_filter, strides= input_strides, padding = input_padding)

def max_pool(input_tensor, inputK_size, input_strides, input_padding ):
    return tf.nn.max_pool(value = input_tensor, ksize = inputK_size, strides= input_strides , padding = input_padding)

class TextCNN(object):
    """
    A convolutional neural network class with one layer. It uses GloVe word embeddings as the convolution filter
    along with max pooling filters of varying sizes (it is parameter set in train script) and rectified linear unit as an
    activation function.

    This class is for a one layer CNN which can classify text into the intended number of classes.
    """

    def __init__(
            self, max_sentence_length,
            num_classes, vocab_size,
            embedding_size, embedding_shape, filter_sizes,
            num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, max_sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # l2 regularization loss initialized
        l2_loss = tf.constant(0.0)


        self.word_embedding = tf.Variable(tf.constant(0.0, shape=embedding_shape), trainable=False)
        self.embedding_placeholder = tf.placeholder(tf.float32, shape = embedding_shape)
        self.embedding_init = self.word_embedding.assign(self.embedding_placeholder)

        # Embedding layer, running on CPU since it is local training and testing
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedded_chars = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create merged convolution and maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                embedding_filter_dim = [filter_size, embedding_size, 1, num_filters]
                weights = init_weights(embedding_filter_dim)
                bias = init_bias(num_filters)

                conv = conv2d(
                            self.embedded_chars_expanded,
                            weights,
                            [1, 1, 1, 1],
                            "VALID"
                )


                # Add activation function to layer (rectified linear unit)
                conv_layer1 = tf.nn.relu( tf.add(conv, bias), name= "conv_layer1_relu")


                # Maxpooling over the outputs
                pooled_max = max_pool(
                               conv_layer1,
                               [1, max_sentence_length - filter_size + 1, 1, 1],
                               [1, 1, 1, 1],
                               "VALID"
                )

                pooled_outputs.append(pooled_max)

        # Combine all the pooled features
        combined_filter_size = num_filters * len(filter_sizes)
        self.combined_pool_outputs = tf.concat(pooled_outputs, 3)
        self.flattened_pool_outputs = tf.reshape(self.combined_pool_outputs, [-1, combined_filter_size])

        # Add dropout to neural net layer
        self.droput_hold_prob = tf.nn.dropout(self.flattened_pool_outputs,  self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions for single layer
        with tf.name_scope("output"):
            weight_full_layer = tf.get_variable(
                "weights",
                shape=[combined_filter_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            bias_full_layer = init_bias(num_classes)
            # l2_loss += tf.nn.l2_loss(weight_full_layer)
            #l2_loss += tf.nn.l2_loss(bias_full_layer)


            # Getting scores for one full layer y = W * x +  b
            self.full_one_scores = tf.nn.bias_add( tf.matmul(self.droput_hold_prob, weight_full_layer), bias_full_layer, name="full_one_scores" )
            self.predictions = tf.argmax(self.full_one_scores, 1, name="predictions")



        # Cross-entropy loss function with L2 Regularization applied
        with tf.name_scope("loss"):
            l2_reg_weights = tf.nn.l2_loss(weight_full_layer)

            # vars   = tf.trainable_variables()
            # all_l2_losses = tf.add_n([ tf.nn.l2_loss(elem) for elem in vars if 'bias' not in elem.name ])

            cross_enthropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.full_one_scores, labels=self.input_y, name="loss")
            self.loss = tf.reduce_mean(cross_enthropy_loss + (l2_reg_weights * l2_reg_lambda))


        with tf.name_scope("prediction_prob"):
                prob = tf.nn.softmax(self.full_one_scores, name="prob")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Number Correct
        with tf.name_scope("num_correct"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, "float"), name = "num_correct")
