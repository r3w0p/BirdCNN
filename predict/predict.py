'''
Created on 30 June 2017

A program that predicts the classification of four different types of bird
based upon the calls they make in audio recordings of them.

@author: pow-pow
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import argv
from numpy import argmax
from functs import *

# Check argument exists for dir root of dataset
if len(argv) < 2:
    print("Argument: <mp3 file>")
    exit()

# List of dir names
FILE_MP3 = argv[1]
HOT_LIST = ["Barn Owl", "Crow", "Oriental Scops Owl", "Western Screech Owl"]
ONE_HOT = len(HOT_LIST)

MY_MODEL = "./model/model.ckpt-1"


# Convolutional Neural Network
with tf.name_scope("cnn"):
    # 'None' and '-1' indicate a variable size. During training, they would be
    # the size of the batch.
    x = tf.placeholder(tf.float32, [None, 65536], name="x")
    x_reshape = tf.reshape(x, [-1, 256, 256, 1], name="x_reshape")

    # Convolution and Pooling
    with tf.name_scope("cp"):

        pool_1 = pool_layer(conv_layer([4, 4, 1, 16], [16], x_reshape, "conv_1"), "pool_1") # -1, 128, 128, 16

        pool_2 = pool_layer(conv_layer([4, 4, 16, 32], [32], pool_1, "conv_2"), "pool_2")   # -1, 64, 64, 32

        pool_3 = pool_layer(conv_layer([4, 4, 32, 64], [64], pool_2, "conv_3"), "pool_3")   # -1, 32, 32, 64

    # Fully-Connected Layer
    with tf.name_scope("fc"):

        # Input (flattened pool)
        # Shape: -1, 65536
        pool_flat = tf.reshape(pool_3, [-1, 32 * 32 * 64], name="pool_flat") # 65536

        # Hidden
        # Weights: 65536, 2048
        # Output:  -1, 2048
        fc_1 = fully_conn(pool_flat, 2048, "fc_1")

        # Readout Layer
        # Weights: 2048, 4
        # Output:  -1, 4
        with tf.name_scope("fc_out"):
            W_fc_out = weight_variable([2048, ONE_HOT]) # 2048, 4
            b_fc_out = bias_variable([ONE_HOT])
            y_conv   = tf.add(tf.matmul(fc_1, W_fc_out), b_fc_out, name="y_conv")
            softmax  = tf.nn.softmax(y_conv) # Use this op to make predictions


# Read in single MP3 file
audio_binary = read_mp3(FILE_MP3)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Restore model for 'cnn' variables
saver = tf.train.Saver()
saver.restore(sess, MY_MODEL)

# Read in audio file
audio = sess.run([audio_binary])

# Feed to y_conv for prediction
prediction = softmax.eval(feed_dict={ x:audio })

# Print prediction
predict_name = HOT_LIST[argmax(prediction)]
predict_prob = round(float(prediction[0][argmax(prediction)]) * 100)

print("\nPredict - %s (%d%%)" % (predict_name, predict_prob))
