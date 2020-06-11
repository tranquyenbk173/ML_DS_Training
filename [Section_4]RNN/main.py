from RNN import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if __name__ == "__main__":
    #Build computation graph
    vocab_path = 'Processed_data/vocab-raw.txt'

    #Try to run, have not tuned hyper-params yet
    train_and_evaluate_RNN(vocab_path, lstm_size=64, batch_size=128)




