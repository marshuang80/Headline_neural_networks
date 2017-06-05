#!/usr/bin/env python
'''
make_plots.py: Make different plots for neural network results

Authorship information:
    __author__ = "Mars Huang"
	__copyright__ = "Copyright 2017, News Headline Project"
	__credits__ = "Dr. Hsu Chun Nan"
    __email__ = "marshuang80@gmail.com:
    __status__ = "Done"
'''
import matplotlib.pyplot as plt
import numpy as np

def graph_training_accuracy(accs, valid_accs):
    '''Graph the accuracies for training and validation

    Args:
        accs (list): list of training accuracies for each epoche
        valid_accs (list): list of validation accuracies for each epoche
    '''
    plt.plot(accs,'g')
    plt.plot(valid_accs, 'b')
    plt.ylabel('% accuracy')
    plt.xlabel('rounds')
    plt.legend(['train_accuracy', 'valid_accuracy'], loc='lower right')
    plt.savefig("training_accuracy")
    plt.close()


def graph_source_accuracy(source_test):
    '''Make bar plots for the average accuracy of each news source

    Args:
        source_test (dictionary): contains the accuracies for each news source
    '''
    # Get average accuracy for each news source
    source_test_result = [[key,sum(val)/len(val)] for key,val in source_test.items()]

    # Sort news sources by their average accuracy
    sorted_source_test = sorted(source_test_result,key = lambda x: x[1])

    # Seperate headline and accuracy
    news_source = [a[0] for a in sorted_source_test]
    real_percentage = [a[1] for a in sorted_source_test]

    # Plot
    y_pos = np.arange(len(news_source))
    plt.bar(y_pos, real_percentage, align='center', alpha=0.5)
    plt.xticks(y_pos, news_source, rotation = 'vertical')
    plt.ylabel('% real')
    plt.title('News sources real news %')
    plt.tight_layout()
    plt.savefig("source_accuracies.png")
    plt.close()
