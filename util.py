import argparse
import sys
import tempfile
import tensorflow as tf
import numpy as np

FLAGS = None
TRAINING = 'policy_training.csv'
TESTING = "policy_testing.csv"

class DataSet():
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(images)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = [self._images[i] for i in perm]
            self._labels = [self._labels[i] for i in perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def str2int(s):
    return int(s)

def read_data_csv(filename):
    features = []
    labels = []
    with open(filename) as inf:
        next(inf)
        for line in inf:
            currentLine = line.strip().split(",")
            currentLine = list(map(str2int, currentLine))
            features.append(currentLine[0:144])
            labels.append(currentLine[144:208])
    return features, labels

def convertOutput(output):
    height = output[0:8]
    width = output[8:16]
    player = output[-1]
    return tf.argmax(height), tf.argmax(width), player

def compareOutputs(out1, out2):
    return tf.logical_and(tf.logical_and(tf.equal(convertOutput(out1)[0], convertOutput(out2)[0]), \
                                         tf.equal(convertOutput(out1)[1], convertOutput(out2)[1])), \
                          tf.equal(convertOutput(out1)[2], convertOutput(out2)[2]))

