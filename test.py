import tensorflow as tf
import util

TRAINING = 'boardState_training.csv'
TESTING = "boardState_testing.csv"

def main():
    training_data, training_target = util.read_data_csv(TRAINING)
    print(training_data[1])
    print(training_target[1])

    sess = tf.InteractiveSession()
    a = util.compareOutputs(training_target[1], training_target[0])
    b = tf.Print(a, [a], message="yeah: ")
    b.eval()

if __name__ == '__main__':
    main()
