import tensorflow.compat.v1 as tf
from utils import *
from MLP import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

if __name__ == "__main__":

    with open('datasets/word_idfs.txt', encoding='utf-8', errors='ignore') as f:
        vocab_size = len(f.read().splitlines())

    mlp = MLP(
        vocab_size = vocab_size,
        hidden_size = 50
    )

    with tf.Session() as sess:
        #load test data
        test_data_reader = DataReader(
            data_path='datasets/test_tf_idfs.txt',
            batch_size=50,
            vocab_size=vocab_size
        )
        #get saved variable:
        epoch = 22 #depends on your saved files
        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)

        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabes_evals = sess.run(
                predicted_labels,
                feed_dict={
                    mlp._X:test_data,
                    mlp._real_Y:test_labels
                }
            )
            matches = np.equal(test_plabes_evals, test_labels)
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader._batch_id == 0:
                break

        print("Epoch: ", epoch)
        print("Accuracy on test data: ", num_true_preds/len(test_data_reader._data))
                #Accuracy on test data:  0.7825278810408922
