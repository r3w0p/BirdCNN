'''
Created on 16 June 2017

A program that trains a Convolutional Neural Network to predict four different
types of bird based upon the calls they make in audio recordings of them.

@author: pow-pow
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import seed, shuffle
from glob import glob
from sys import argv
from os import path, makedirs
import numpy as np

from functs import *

# Check argument exists for dataset folder
if len(argv) < 2:
    logme("Argument: <bird dataset directory>")
    exit()

ROOT     = argv[1] # Dataset
HOT_LIST = ["BarnOwl", "Crow", "OSOwl", "WSOwl"] # Valid one-hot classes
ONE_HOT  = len(HOT_LIST)


# For logging progress to file
def logme(text):
    save = open("log.txt", 'a')
    save.write(text + '\n')
    save.close()

# Generate one-hot variable for audio clip
def get_one_hot(audio_paths):
    one_hot = []

    for path in audio_paths:
        # Get folder name (should match value in HOT_LIST)
        class_type = str(path).split('/')[-2]

        label = -1
        for i in range(0, ONE_HOT):
            if class_type == HOT_LIST[i]:
                label = i
                break

        # If type doesn't match valid one-hot type, error
        if label == -1:
            logme("ERROR generating one-hot for path %s." % str(path))
            exit()

        # Create one-hot
        hot = [0] * ONE_HOT
        hot[label] = 1
        one_hot.append(hot)

    return np.array(one_hot)


# Constants for running training and testing
SIZE_TOTAL = 7130 # All training examples

SIZE_TRAIN = 6500
SIZE_TEST  = SIZE_TOTAL - SIZE_TRAIN # 630

SIZE_BATCH_TRAIN = 50
SIZE_BATCH_TEST  = 30

TRAIN_ITER = int(SIZE_TRAIN / SIZE_BATCH_TRAIN) # 6500 / 50
TEST_ITER  = int(SIZE_TEST  / SIZE_BATCH_TEST)  # 630 / 30

EPOCH = 1


# Saver to store final model
SAVER_DIR   = "model"
SAVER_FINAL = SAVER_DIR + "/model.ckpt"

if not path.exists(SAVER_DIR):
    makedirs(SAVER_DIR)


# Collate mp3 file locations and shuffle
ALL_FILES = list()

for hot in HOT_LIST:
    ALL_FILES += glob("./%s/%s/*.mp3" % (ROOT, hot))

seed(42)
shuffle(ALL_FILES)


# Queue and batching ops
audio_queue = tf.train.string_input_producer(ALL_FILES, shuffle=False, capacity=SIZE_TOTAL)
batch_size = tf.placeholder(tf.int32)
batch_examples, batch_paths = input_pipeline(audio_queue, batch_size, SIZE_TOTAL)


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


# Loss and Optimizer
with tf.name_scope("loss"):
    y_ = tf.placeholder(tf.float32, [None, ONE_HOT], name="y_")
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name="cross_entropy")
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy
with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


logme("Init")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# Only save variables from scope 'cnn' for reuse in predictions
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cnn"))

logme("Coord")
# Start populating the queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

logme("Start")
for i in range(0, EPOCH):

    logme("Epoch: %s / %s" % (str(i), str(EPOCH)))

    # Train
    # (ensure SIZE_BATCH_TRAIN divides *perfectly* into SIZE_TRAIN)
    logme("TRAINING")
    for i in range(0, TRAIN_ITER):
        e_batch, p_batch = sess.run([batch_examples, batch_paths], feed_dict={ batch_size:SIZE_BATCH_TRAIN })
        l_batch = get_one_hot(p_batch)

        logme(" > " + str(i) + ", Train: " + str(SIZE_BATCH_TRAIN))
        train_step.run(feed_dict={ x:e_batch, y_:l_batch })

    # Test
    # (ensure SIZE_BATCH_TEST divides *perfectly* into SIZE_TEST)
    # (too much memory required to store all test audio, so batch and take average)
    logme("TESTING")
    test_sum = 0
    for i in range(0, TEST_ITER):
        e_batch_test, p_batch_test = sess.run([batch_examples, batch_paths], feed_dict={ batch_size:SIZE_BATCH_TEST })
        l_batch_test = get_one_hot(p_batch_test)

        acc = accuracy.eval(feed_dict={ x:e_batch_test, y_:l_batch_test })
        logme(" > " + str(acc))
        test_sum += acc

    test_accuracy = test_sum / TEST_ITER
    logme(" > Accuracy: %f" % test_accuracy)


# Final Model
logme("Final Model: " + SAVER_FINAL)
save_final = saver.save(sess, SAVER_FINAL, global_step=EPOCH)

logme("Stop")
coord.request_stop()
coord.join(threads)

logme("FINISHED")
