'''
Created on 16 June 2017

Additional functions used to train the model.

@author: pow-pow
'''

import tensorflow as tf

def read_single(filename_queue):
    """Takes a queue of audio file locations and reads a single one."""
    reader = tf.WholeFileReader()
    audio_path, audio_binary = reader.read(filename_queue)

    # audio roughly 2 seconds in duration
    waveform = tf.contrib.ffmpeg.decode_audio(
        audio_binary,
        file_format='mp3',
        samples_per_second=44100,
        channel_count=1)

    # 65536 to achieve shape of 256x256x1, a convenient shape for
    # the convnet as it can be continually halved down to 1
    reshaped = tf.reshape(waveform, [-1])
    example = tf.slice(reshaped, [0], [65536])

    # normalize to 0-1
    example_norm = tf.div(
        tf.subtract(
            example,
            tf.reduce_min(example)
        ),
        tf.subtract(
            tf.reduce_max(example),
            tf.reduce_min(example)
        )
    )

    return example_norm, audio_path


def input_pipeline(filename_queue, batch_size, capacity):
    """Batches audio files."""
    example, path = read_single(filename_queue)
    return tf.train.batch([example, path], batch_size, capacity=capacity)


#
# VARIABLE INIT
#
def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return initial

def bias_variable(shape):
    initial = tf.Variable(tf.constant(0.1, shape=shape))
    return initial


#
# CONV NET
#
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(w_, b_, x_, name):
    with tf.name_scope(name):
        W_conv = weight_variable(w_)
        b_conv = bias_variable(b_)
        conv = tf.nn.relu(conv2d(x_, W_conv) + b_conv)
        return conv

def pool_layer(conv, name):
    with tf.name_scope(name):
        pool = max_pool_2x2(conv)
        return pool


#
# FULLY-CONNECTED LAYER
#
def fully_conn(X, n_neurons, name):
    with tf.name_scope(name):
        # Weights: inputs x neurons
        W_fc = weight_variable([int(X.get_shape()[1]), n_neurons])
        b_fc = bias_variable([n_neurons])
        fc = tf.nn.relu(tf.matmul(X, W_fc) + b_fc)
        return fc
