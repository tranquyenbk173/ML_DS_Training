import tensorflow.compat.v1 as tf
from MLP import *
from utils import *

if __name__ == "__main__":

    ## Create a computation graph:
    with open('datasets/word_idfs.txt', encoding='utf-8', errors='ignore') as f:
        vocab_size = len(f.read().splitlines())

    mlp = MLP(
        vocab_size = vocab_size,
        hidden_size = 50
    )
    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss = loss, learning_rate=0.1)

    ##Open a session to run and write params to file
    with tf.Session() as sess:
        train_data_reader = DataReader(
            data_path='datasets/train_tf_idfs.txt',
            batch_size=50,
            vocab_size=vocab_size
        )
        step, MAX_STEP = 0, 1001

        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            plabels_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict = {
                    mlp._X:train_data,
                    mlp._real_Y:train_labels
                }
            )
            #print("loss eval = ", loss_eval)

            step +=1
            #print('step: {}, loss: {}'.format(step, loss_eval))

            #if step % 100 == 0:
                #print("Write step = {}, epoch = {}".format(step, e))
        trainable_variables = tf.trainable_variables()
        print("Save params!")
        for variable in trainable_variables:
            save_parameters(
                name = variable.name,
                value = variable.eval(),
                epoch = train_data_reader._num_epoch
            )


    ##evaluate with test set
    with tf.Session() as sess:
        test_data_reader = DataReader(
            data_path='datasets/test_tf_idfs.txt',
            batch_size=50,
            vocab_size=vocab_size
        )
        epoch = train_data_reader._num_epoch
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
