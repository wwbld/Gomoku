import argparse
import sys
import tempfile
import tensorflow as tf
import numpy as np

FLAGS = None
TRAINING = 'boardState_training.csv'
TESTING = "boardState_testing.csv"

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
            features.append(currentLine[0:129])
            labels.append(currentLine[129:147])
    return features, labels

def convertOutput(output):
    height = output[0:8]
    width = output[8:16]
    player = output[-1]
    return height.index(1), width.index(1), player

