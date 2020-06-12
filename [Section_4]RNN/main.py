from RNN import *
from utils import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if __name__ == "__main__":

    vocab_path = 'Processed_data/vocab-raw.txt'
    train_path = 'Processed_data/20news-train-encoded.txt'
    test_path = 'Processed_data/20news-test-encoded.txt'

    #Try to get best_hyper_params:
    list_values = [[32, 64], [50, 50], [64, 64], [64, 128], [128, 256]]
        #list_values = [[32, 64]]
    best_values = get_best_hyper_param(vocab_path, train_path, list_values)

    #Run run run
    lstm_size, batch_size = best_values[0], best_values[1]
    print("Best values: lstm_size", lstm_size, "- batch_size", batch_size)
    train_and_evaluate_RNN(vocab_path, train_path, test_path,
                            lstm_size=lstm_size, batch_size=batch_size)




