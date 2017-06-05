#!/usr/bin/env python
'''
data_generator.py: Contains funtions that generates data for neural networks

Authorship information:
    __author__ = "Mars Huang"
	__copyright__ = "Copyright 2017, News Headline Project"
	__credits__ = "Dr. Hsu Chun Nan"
    __email__ = "marshuang80@gmail.com:
    __status__ = "Done"
'''

import numpy as np
import random
from input_processor import word_2_vec, load_data, clean

def shuffle_data(x,y,seed):
    '''Shuffle the input data

    Args:
        x (list): list of news headlines
        y (list): list of one-hot encoded labels
		seed (int): seed for random generator

    Return:
        x (list): list of shuffled news headlines
        y (list): list of shuffled one-hot encoded labels
    '''
    print("Shuffling data...")
    np.random.seed(seed)
    idx = np.random.permutation(np.arange(len(y)))
    y = np.array(y)
    return x[idx], y[idx]


def generate_data(real_headlines, fake_headlines):
    '''Generate cleaned, shuffled and vectorized input data

    Args:
        real_headlines (string): Text file that contains real news headlines
        fake_headlines (string): Text file that contains real news headlines

    Return:
        x (list): list of vectorized headlines
        y (list): list one-hot encoded labels
        processor (VocabularyProcessor): Processor to turn sentences to vector
    '''
    # Load data from input files
    x,y = load_data(real_headlines, fake_headlines)

    # Transform sentences to vectors
    processor, x = word_2_vec(x)

    # Shuffle Training data
    x,y = shuffle_data(x,y,7)

    return x,y,processor


def test_batch(x,y,ratio):
    '''Split the input data into train and test set

    Args:
        x (list): vectorized headlines
        y (list): one-hot encoded labels
        ratio (float): ratio of training to testing set

    Return:
        x_train: batch of train data
        y_train: batch of train label
        x_test: batch of test data
        y_test: batch of test label
    '''
    split = int(len(x) * ratio)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    return x_train, y_train, x_test, y_test


def batch(x,y,batch_size, ratio):
    '''Ge nerate random batches of samples during CNN training

    Args:
        x (list): vectorized headlines
        y (list): one-hot encoded labels
        batch_size (int): size of each training batch
        ratio (float): ratio of training to validation set

    Return:
        x_train: batch of train data
        y_train: batch of train label
        x_valid: batch of validation data
        y_valid: batch of validation label
     '''
    # Shuffle data
    idx = random.sample(range(len(x)),batch_size)
    xs, ys = x[idx], y[idx]

    # Generate batches
    split = int(len(xs) * ratio)
    x_train, y_train = xs[:split], ys[:split]
    x_valid, y_valid = xs[split:], ys[split:]

    return x_train, y_train, x_valid, y_valid


def real_dictionary_generator(real_with_source):
    '''Generates a dictionary with news source as key and headlines as variables\
    for all real news headlines.

    Args:
        real_with_source (string): the name of the file containing all real news

    Return:
        real_dict (dictionary): Dictionary with news source as key and headlines\
                                as variables for all real news headlines
        real_headlines (list): all real news headlines
    '''
    # Read in file contents
    inputs = [a.strip('\n').split('\t') for a in open(real_with_source,"r",encoding='utf8')]

    # Format input: (cleaned headlines, one-hot encoded label, news source)
    inputs_formatted = [[clean(a[0]),[0,1], a[1]] for a in inputs if len(a) == 2]

    # Limit headline length between 5 and 20 words
    inputs_formatted = [d for d in inputs_formatted if len(d[0].split()) <= 20 \
                        and len(d[0].split()) >= 4]

    # Get all headlines
    real_headlines = [h[0] for h in inputs_formatted]

    # Generate dictionary
    real_dict = {}
    for data in inputs_formatted:
        if data[2] not in real_dict:
            real_dict[data[2]] = [[data[0],data[1]]]
        else:
            real_dict[data[2]].append([data[0],data[1]])

    return real_dict, real_headlines


### Functions for RNN generator ###
def num_2_word_generator(processor):
    '''Return dictonaries that converts num to word and word to num

    Args:
        processor (Vocabulary Processor): Processor that vectoize headlines

    Returns:
        word_2_num (dictonary): dictonary that maps words to numbers
        num_2_word (dictonary): dictionary that maps numbers to words
    '''

    word_2_num = processor.vocabulary_._mapping
    num_2_word = {v:k for k,v in word_2_num.items()}

    return word_2_num, num_2_word


def one_hot(sentence, vocab_size):
    '''Turn input sentence input a list of one-hot encoded vocabs.

    Input can also be a numeric representation of word during the start of \
    generating headlines.

    Args:
        sentence (list): list of vecotized headline
        sentence (int): can also be just a int representation of word
        vocab_size (int): size of the vocab
    Returns:
        vec (list): one-hot encoded sentence
    '''
    # Check if input is sentence
    if type(sentence) != int:
        vec = np.zeros((len(sentence), vocab_size))
        for idx, word in enumerate(sentence):
                vec[idx, word] = 1
    # If input is just one word
    else:
        vec = np.zeros((1,20,vocab_size))
        vec[0,0,sentence] = 1

    return vec


def shift_y(sentence):
    '''Remove the first word of the sentencce and add a space at the end
    This used as the target for RNN. Each word's label is the next word in \
    the sentence

    Args:
        sentence (list): List of vectoized sentnece

    Returns:
        y_list (numpy_array): shifted sentence
    '''
    y_list = list(sentence)[1:]
    y_list.append(0)
    return np.array(y_list)


def rnn_batch(batch_size, x, vocab_size):
    '''Generate a batch of rnn inputs and labels

    Args:
        batch_size (int): size of the batch
        x (list): vecotized input
        vocab_size (int): size of the vocab

    Returns:
        x_batch (list): A list of one-hot encoded input
        y_batch (list): A list of one-hot enocded labels (shifted x)
    '''
    batch_idx = random.sample(range(len(x)), batch_size)
    x_batch = np.array([one_hot(x[idx],vocab_size) for idx in batch_idx])
    y_batch = np.array([one_hot(shift_y(x[idx]),vocab_size) for idx in batch_idx])
    return x_batch, y_batch

