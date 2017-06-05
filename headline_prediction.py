#!/usr/bin/env python
'''
headline_predictiton.py: Convolutional Neural Network that classifies real and \
						 fake news based on their headlines.

Authorship information:
    __author__ = "Mars Huang"
	__copyright__ = "Copyright 2017, News Headline Project"
	__credits__ = "Dr. Hsu Chun Nan"
    __email__ = "marshuang80@gmail.com:
    __status__ = "Done"
'''

import numpy as np
import tensorflow as tf
import sys
from data_generator import *
from make_plots import *
from tensorflow.contrib import learn


# Variables
real_news = 'real.txt'
fake_news = 'fake.txt'
real_with_source = 'real_with_source.txt'
train_to_test_ratio = 0.95
accs,valid_accs,test_accs = [],[],[]

# CNN hypterparameters
sen_len = 20
embedding_size = 128 #standard
num_filters = 128
filter_size = [1,2,3]
epoche = 10
alpha = 5e-3

# Input placeholders
x_input = tf.placeholder(tf.int32,[None, sen_len])
y_input = tf.placeholder(tf.float32,[None, 2])

def word_enbedding(x_input):
    '''Apply word enbedding to vectorized input headlines

    Args:
        x_input (tensor): vectorized input headlines

    Return:
        embedded_input (tensor): embedded headlines
    '''
    #Word embeding layer
    with tf.name_scope("word_embedding"):
        word2vec = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))
        embedded = tf.nn.embedding_lookup(word2vec,x_input)

        #CNN requires input of shape [batch, width, height, channel]
        #No channel thus put 1 for last dimention
        embedded_input = tf.expand_dims(embedded,-1)

    return embedded_input


def conv_layer(embedded_input):
    '''The convolutional neural network layers with different filter size

    Args:
        embedded_input (tensor): embedded headlines

    Return:
        conv_outputs (list): lists of convolutional layer output tensors
    '''
    conv_outputs  = []

    # Iteration through different filter size
    for size in (filter_size):
        with tf.name_scope("conv%s"%(str(size))):
            # Weight and bias
            conv_w = tf.Variable(tf.truncated_normal([size,embedding_size,1, num_filters],stddev = 0.1))
            b = tf.Variable(tf.constant(0.1,shape=[num_filters]))

            # Convolution
            conv = tf.nn.conv2d(embedded_input, conv_w, strides = [1,1,1,1], padding = "VALID")

            # Activation:
            out = tf.nn.relu(tf.nn.bias_add(conv,b), name="activation")

            # Max pooling
            pool = tf.nn.max_pool(out,ksize = [1,sen_len - size + 1, 1, 1], strides = [1,1,1,1], padding = 'VALID')
        conv_outputs.append(pool)

    return conv_outputs


def flatten(conv_outputs):
    '''Combine convolutional outputs and apply dropouts

    Args:
        conv_outputs (list): lists of convolutional layer output tensors

    Return:
        drop_layer (tensor): tensor after applying dropouts
    '''
    # Combine pooled layer by concatanate on the 3rd dimention (filters)
    pooling = tf.concat(conv_outputs,3)
    pool_flatten = tf.reshape(pooling, [-1, num_filters * len(filter_size)])

    # Apply dropouts
    with tf.name_scope("dropout"):
        drop_layer = tf.nn.dropout(pool_flatten,0.5)

    return drop_layer


def output_prediction(drop_layer):
    '''Predict headline as real or fake, compute cost and accuracy, and optimize

    Args:
        drop_layer (tensor): tensor after applying dropouts

    Return:
        output (list): confidence in fake and real label
        predictions (list): prediction of input headlines
        accuracy (float): the percent accuracy of how many labels are predicted\
                        correctly
        optimizer (Adam): the optimizer for the network
    '''
    # Compute output with softmax
    with tf.name_scope("output"):
        W = tf.get_variable("W",shape = [num_filters*len(filter_size),2],initializer=tf.contrib.layers.xavier_initializer())
        B = tf.Variable(tf.constant(0.1,shape=[2]), name = 'b')
        output = tf.nn.softmax(tf.matmul(drop_layer,W)+B)
        predictions = tf.argmax(output,1)

    # Calculate cross-entropy loss
    with tf.name_scope("loss"):
        losses = tf.reduce_mean(-tf.reduce_sum(y_input*tf.log(output),reduction_indices=[1]))

    # Calculate prediction accuracy
    with tf.name_scope("accuracy"):
        pred = tf.equal(predictions,tf.argmax(y_input, 1))
        accuracy = tf.reduce_mean(tf.cast(pred,"float"))

    # Optimize network
    optimizer = tf.train.AdamOptimizer(alpha).minimize(losses)

    return output, predictions, accuracy, optimizer


def train_CNN():
    '''Train the convolutiona l neural network

    Return:
        accs (list): a list of training accuracy during each epoche
        valid_accs (list): a list of validation accuracy for each epoche
    '''
    print("start training...")
    for i in range(epoche):

        # Generate batch
        x_batch, y_batch, x_valid, y_valid = batch(x,y, int(len(x)/4),0.9)

        # Feed input dictionary for train and validation
        feed_dict = {
                x_input: x_batch,
                y_input: y_batch,
            }
        feed_dict_valid = {
                x_input: x_valid,
                y_input: y_valid,
            }

        # Compute accuracy
        acc, _ = sess.run([accuracy,optimizer],feed_dict = feed_dict)
        acc_valid = sess.run(accuracy,feed_dict = feed_dict_valid)

        # Append accuracies
        accs.append(acc)
        valid_accs.append(acc_valid)

        print("epoche: %i    train: %f    valid: %f"%(i,acc,acc_valid))

    #saver = tf.train.Saver()
    #saver.save(sess, "/tmp/model.ckpt")
    return accs, valid_accs


def test_CNN():
    '''Test CNN on test set

    Return:
        acc_test (float): test accuracy
    '''
    # Feed input dictionary for testing
    feed_dict_valid = {
            x_input: x_test,
            y_input: y_test,
        }

    # Compute test accuracy
    acc_test = sess.run(accuracy,feed_dict = feed_dict_valid)

    return acc_test


def test_CNN():
    '''Test CNN on test set

    Return:
        acc_test (float): test accuracy
    '''
    # Feed input dictionary for testing
    feed_dict_valid = {
            x_input: x_test,
            y_input: y_test,
        }

    # Compute test accuracy
    acc_test = sess.run(accuracy,feed_dict = feed_dict_valid)

    print("Test accuracy: %f"%(acc_test))

    return acc_test


def test_news_source(processor):
    '''Test the accuracy of predicting each news source

    Args:
        processor (VocabularyProcessor): Processor that vecrorize headlines

    Returns:
        source_test (Dictionary): accuracies for each news source
    '''
    real_dict, real_headlines = real_dictionary_generator(real_with_source)
    source_test = {}

    # Run 10 predictions and take average
    for i in range(10):

        for key in real_dict:
            # Generate testing data for news source
            X = np.array([a[0] for a in real_dict[key]])
            Y = np.array([a[1] for a in real_dict[key]])
            X,Y,_,_ = batch(X,Y,10000,1.0)
            X = np.array(list(processor.fit_transform(X)))

            # Feed input dictionary
            feed_dict_source = {
                    x_input: X,
                    y_input: Y,
                }

            # Test source accuracy
            acc_source = sess.run(accuracy,feed_dict_source)

            # Store result in dictionary
            if key in source_test:
                source_test[key].append(acc_source)
            else:
                source_test[key] = [acc_source]

    return source_test


if __name__ == "__main__":
    # Load files and inputs
    x,y,processor = generate_data(real_news,fake_news)
    x,y,x_test,y_test = test_batch(x,y,train_to_test_ratio)
    vocab_size = len(processor.vocabulary_)

    # Build network
    print("Building network....")
    embedded_input = word_enbedding(x_input)
    conv_outputs = conv_layer(embedded_input)
    drop_layer = flatten(conv_outputs)
    output, predictions, accuracy, optimizer = output_prediction(drop_layer)

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train CNN
        accs, valid_accs = train_CNN()
        graph_training_accuracy(accs, valid_accs)
        print("Training max accuracy: %f"%(max(accs)))
        print("Validation max accuracy: %f"%(max(valid_accs)))

        # Test CNN
        acc_test = test_CNN()

        # Test news sources
        if len(sys.argv) > 1 and sys.argv[1] == "-s":
            source_test = test_news_source(processor)
            graph_source_accuracy(source_test)
