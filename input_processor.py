#!/usr/bin/env python
'''
input_processor.py: Load, clean and process input text data into vectors.

Authorship information:
    __author__ = "Mars Huang"
	__copyright__ = "Copyright 2017, News Headline Project"
	__credits__ = "Dr. Hsu Chun Nan"
    __email__ = "marshuang80@gmail.com:
    __status__ = "Done"
'''

import csv
import numpy as np
import random
import re
from tensorflow.contrib import learn

def clean(string):
    '''Clean the input headlines by removing unwanted words and symbols

    Args:
        string (String): input headline

    Return:
        string (String): cleaned and processed headline

    '''
    try:
        # Manually remove repetitive and unwanted words
        string = string.replace('apos','')
        string = string.replace('PHOTOS','')
        string = string.replace('(VIDEO)','')
        string = string.replace('[VIDEO]','')
        string = string.replace('[Video]','')
        string = string.replace('Russia News Now','')
        string = string.replace('TruthFeed','')
        string = string.replace('Guardian Liberty Voice','')
        string = string.replace('LIBERTY WRITERS NEWS','')
        string = string.replace('Conservative Daily Post','')
        string = string.replace("The Onion - America's Finest News Source",'')
        string = string.replace('minutes ago','')
        string = string.replace('minute ago','')
        string = string.replace('seconds ago','')
        string = string.replace('second ago','')
        string = string.replace('hour minutes ago','')
        string = string.replace('hours minutes ago','')
        string = string.replace('hour ago','')
        string = string.replace('hours ago','')
        string = string.replace('hour','')
        string = string.replace('Shares','')
        string = string.replace('Live Update','')
        # Remove all uncommon characters
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

        # Remove trailing random numbers, date and time
        while string[-1] in "AMP0123456789 ,s":
            string = re.sub("[AM PM A M s]+$","",string)
            string = re.sub("[ \t]+$","",string)
            string = re.sub("[0-9]+$","",string)
            string = re.sub("[,]+$","",string)
        string = re.sub(r"\'s", "\'s", string)
        string = re.sub(r"\'ve", "\'ve", string)
        string = re.sub(r"n\'t", "\'t", string)
        string = re.sub(r"\'re", "\'re", string)
        string = re.sub(r"\'d", "\'d", string)
        string = re.sub(r"\'ll", "\'ll", string)

        # Remove random spaces
        string = re.sub("^\s*","",string)
        string = re.sub("  +"," ",string)

        # Format puctuations
        '''
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ",string)
        string = re.sub(r"\?", " \? ",string)
        string = re.sub(r"\s{2,}", " ",string)
        '''
    except Exception: return ''
    return string


def word_2_vec(x):
    '''Transforming input headlines to vectors

    Args:
        x (list[string]): list of news headlines

    Returns:
        processor (VocabularyProcessor): Processor that vectorize headlines
        x_vec (list[int]): vectorized headlines
    '''
    print("Transforming sentences to vectors...")
    processor = learn.preprocessing.VocabularyProcessor(20) # Sentence size 20
    x_vec = np.array(list(processor.fit_transform(x)))

    return processor, x_vec


def load_data(real_headlines,fake_headlines):
    ''' Load headlines from the real and fake news text file and process it

    Args:
        real_headlines (string): Text file that contains real news headlines
        fake_headlines (string): Text file that contains real news headlines

    Returns:
        x (list[string]) : List of lists of cleaned and processed headlines
        y (list[int]) : List of one-hot encoded label of real or fake news

    '''
    # Reading in real and fake news from the file
    print("Reading input data...")
    pos = list(csv.reader(open(real_headlines,'r',encoding='utf8'),delimiter='\t'))
    neg = list(csv.reader(open(fake_headlines,'r',encoding='utf8'),delimiter='\t'))

    # Process/clean the headline and limit length of headlines between 5 to 20
    print("Processing data...")
    pos = [clean(p[0]) for p in pos]
    pos = [a for a in pos if (len(a.split()) <= 20 and len(a.split()) >=5)]
    neg = [clean(n[0]) for n in neg]
    neg = [b for b in neg if (len(b.split()) <= 20 and len(b.split()) >=5)]

    # Randomly sample real news to have equal amount of training data
    idx = random.sample(range(len(pos)), len(neg))
    pos = [pos[i] for i in idx]
    x = pos + neg

    # Make one-hot encoded label
    p_label = [[0,1] for p in pos]
    n_label = [[1,0] for n in neg]
    y = p_label + n_label

    return x, y



