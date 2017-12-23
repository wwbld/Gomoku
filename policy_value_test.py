import tempfile
import tensorflow as tf
import numpy as np
import util

FLAGS = None
POLICY_TRAINING = 'data/policy_training.csv'
POLICY_TESTING = 'data/policy_testing.csv'
VALUE_TRAINING = 'data/value_training.csv'
VALUE_TESTING = "data/value_testing.csv"

def main():
    policy_testing_data, policy_testing_target = util.read_policy_csv(POLICY_TESTING)
    value_testing_data, value_testing_target = util.read_value_csv(VALUE_TESTING)

    policy_graph = util.ImportGraph('policy_model/', 'policy_model')
    value_graph = util.ImportGraph('value_model/', 'value_model') 
   
    policy_graph.get_accuracy(policy_testing_data, policy_testing_target)
    value_graph.get_accuracy(value_testing_data, value_testing_target)  

    policy_graph.get_predict(policy_testing_data[0])
    value_graph.get_predict(value_testing_data[0])

if __name__ == '__main__':
    main()
             
