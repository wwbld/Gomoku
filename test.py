import tensorflow as tf
import util

TRAINING = 'boardState_training.csv'
TESTING = "boardState_testing.csv"

def main():
    training_data, training_target = util.read_data_csv(TRAINING)
    print(training_data[1])
    print(training_target[1])
    print(util.convertOutput(training_target[1])[0])
    print(util.convertOutput(training_target[1])[1])
    print(util.convertOutput(training_target[1])[2])

if __name__ == '__main__':
    main()
