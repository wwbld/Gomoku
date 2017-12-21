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
TRAINING = 'data/value_training.csv'
TESTING = "data/value_testing.csv"

def main():
    testing_data, testing_target = util.read_value_csv(TESTING)
    test = util.DataSet(testing_data, testing_target)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("value_model/value_model.meta")
        saver.restore(sess, tf.train.latest_checkpoint("value_model/"))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        output = graph.get_tensor_by_name("accuracy/output:0")
        target = graph.get_tensor_by_name("accuracy/target:0")
        keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
        feed_dict = {x:testing_data, y_:testing_target, keep_prob:1.0}
        predict_op = graph.get_tensor_by_name("predict_op:0")
        print("accuracy is %g" % sess.run(predict_op, feed_dict))     
        #print("output is {0}".format(sess.run([output], feed_dict)))
        #print("target is {0}".format(sess.run([target], feed_dict)))



if __name__ == '__main__':
    main() 
