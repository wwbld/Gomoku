#from __fyture__ import absolute_import
#from __future__ import division
#from __future__ import print_fuction

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import util

FLAGS = None
TRAINING = 'policy_training.csv'
TESTING = "policy_testing.csv"

def main():
    testing_data, testing_target = util.read_data_csv(TESTING)
    test = util.DataSet(testing_data, testing_target)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("my_model.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./"))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
        feed_dict = {x:[testing_data[0]], y_:[testing_target[0]], keep_prob:1.0}
        predict_op = graph.get_tensor_by_name("predict_op:0")
        print("accuracy is %g" % sess.run(predict_op, feed_dict))     



if __name__ == '__main__':
    main() 
