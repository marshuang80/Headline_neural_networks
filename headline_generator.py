#!/usr/bin/env python
'''
headline_generator.py: Using RNN to learn and generate news headlines based on \
the input text file.
Authorship information:
    __author__ = "Mars Huang"
	__copyright__ = "Copyright 2017, News Headline Project"
	__credits__ = "Dr. Hsu Chun Nan"
    __email__ = "marshuang80@gmail.com:
    __status__ = "Need improvement"
TODO:
    Need to fine-tune RNN to generate headlines that makes sense
'''
# Imports
import numpy as np
import random
import tensorflow as tf
from data_generator import *

# Variables
real_news = 'real.txt'
fake_news = 'fake.txt'
x,y,processor = generate_data(real_news,fake_news)
word_2_num, num_2_word = num_2_word_generator(processor)
vocab_size = len(processor.vocabulary_)
costs = []

# RNN model parameters
learning_rate = 0.05
lstm_size = 256
num_layers = 1
input_size = out_size = vocab_size
epoche = 2000
batch_size = 128
sentence_len = 20
start_word = "You"

# Rnn Weight and bias
W = tf.Variable(tf.random_normal((lstm_size, out_size), stddev=0.01))
B = tf.Variable(tf.random_normal((out_size, ), stddev=0.01))

# Input placeholders
x_input = tf.placeholder(tf.float32, shape = (None, None, input_size))
y_input = tf.placeholder(tf.float32, shape = (None, None, out_size))
lstm_init = tf.placeholder(tf.float32, shape = (num_layers*2*lstm_size))


def rnn(x_input, lstm_init, W, B):
    '''Recurrent neural network layer of the neural network

    Args:
        x_input (list): vectorized headlines
        lstm_init (lstm state): initial state of lstm cells # Not used
        W (tf.Variable): weights for RNN
        B (tf.Variable): bias for RNN

    Return:
        outputs (list): RNN output
        network_output (list): RNN output after weights and bias
        lstm_new (lstm): lstm state after each epoche
    '''
    x_input = tf.unstack(x_input,num = sentence_len, axis = 1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple = False)
    outputs, lstm_new = tf.contrib.rnn.static_rnn(lstm_cell, x_input, dtype=tf.float32, scope='rnn')
    output_reshape = tf.reshape( outputs, [-1, lstm_size])
    network_output = tf.matmul(output_reshape, W) + B
    return outputs, network_output, lstm_new


def optimization(outputs, network_output, y_input):
    '''Optimization for recurrent neural network

    Args:
        outputs (list): RNN output
        network_output (list): RNN output after weights and bias
        lstm_new (lstm): lstm state after each epoche

    Return:
        loss (float): the neural network loss
        optimizer (RMSPropOptimizer): RMSProp Optimizer
        final_output (list): confidence for each output word
     '''
    shape_ = tf.shape(outputs)
    final_output = tf.reshape(tf.nn.softmax(network_output), (shape_[0],shape_[1], out_size))
    y_input_reshaped = tf.reshape(y_input, [-1,out_size])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_input_reshaped))
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss)
    return loss, optimizer, final_output


def generate_headlines(start_word, final_output, sess):
    '''Generate headlines from RNN

    Args:
        stard_word (int): the numeric representation of the first word of the \
                        sentence
        final_output (list): confidence for each output word
        sess (tf.sess): tensorflow session

    Return:
        out_sentence (list): the generated sentence from RNN
    '''
    out_sentence = [num_2_word[start_word]]
    start_word_vec = one_hot(int(start_word),vocab_size)
    output_confidence = sess.run(final_output, feed_dict = {x_input:start_word_vec})
    for j in range(sentence_len-1):
        next_word = np.random.choice(range(vocab_size),p=output_confidence[j][0])
        #next_word = np.argmax(out[j][0])
        start_word_vec[0][j] = next_word
        out_sentence.append(num_2_word[next_word])
        output_confidence = sess.run(final_output, feed_dict = {x_input: start_word_vec})
    return out_sentence


def train_RNN():
    '''Train the reuccurent neural network model, and output generate headlines\
    every 50 epche
    '''
    # Build neural network
    outputs, network_output, lstm_new = rnn(x_input, lstm_init, W, B)
    loss, optimizer, final_output = optimization(outputs, network_output, y_input)

    # Run neural network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        for i in range(epoche):
            x_batch, y_batch = rnn_batch(batch_size,x,vocab_size)
            train_dict = {x_input:x_batch, y_input:y_batch}
            cost, lstm_state, _ = sess.run([loss, lstm_new ,optimizer], feed_dict=train_dict)
            costs.append(cost)

            # Test generating headlines
            if i % 50 == 0:
                start_word = np.argmax(x_batch[0][0])
                out_sentence = generate_headlines(start_word, final_output, sess)
                print("epoche: %i, loss: %f"%(i,cost))
                print(' '.join(out_sentence))

if __name__ == '__main__':
    train_RNN()
