import tensorflow as tf
import util

TRAINING = 'data/value_training.csv'
TESTING = "data/value_testing.csv"

def main():
    training, testing = util.read_value_csv(TESTING)
    print(testing)

if __name__ == '__main__':
    main()
