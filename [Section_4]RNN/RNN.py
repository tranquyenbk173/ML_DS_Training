import tensorflow as tf
import numpy as np

class RNN:
    def __init__(self, vocab_size, embedding_size, lstm_size, batch_size,
                 pretrained_w2w_path = None, MAX_DOC_LENGTH=500):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._batch_size = batch_size
        self._lstm_size = lstm_size

        self._data = tf.placeholder(tf.int32, shape = [batch_size, MAX_DOC_LENGTH])
        self._labels = tf.placeholder(tf.int32, shape=[batch_size, ])
        self._sentence_lengths = tf.placeholder(tf.int32, shape=[batch_size, ])
        self._final_tokens = tf.placeholder(tf.int32, shape=[batch_size, ])


    def embedding_layer(self, indices):
        pretrained_vectors = []
        pretrained_vectors.append(np.zeros(self._embedding_size))
        np.random.seed(2020)

        for _ in range(self._vocab_size + 1):
            pretrained_vectors.append(np.random.normal(loc=0, scale=1.,
                                                       size = self._embedding_size))
        pretrained_vectors = np.array(pretrained_vectors)    
        self._embedding_matrix = tf.get_variable(
            name = 'embedding',
            shape = (self._vocab_size + 2, self._embedding_size),
            initializer = tf.constant_initializer(pretrained_vectors)
        )
        
        return tf.nn.embedding_layer(self._embedding_matrix, indices)


    def LSTM_layer(self, embeddings):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
        zero_state = tf.zeros(shape = (self._batch_size, self._lstm_size))
        initial_state = tf.contrib.rnn.LSTMStateTuple(zero_state, zero_state)

        lstm_inputs = tf.unstack(
            tf.transpose(embeddings, perm=[1, 0, 2])
        )

        lstm_outputs, last_state = tf.nn.static_rnn(
            cell = lstm_cell,
            inputs = lstm_inputs,
            initial_state=initial_state,
            squence_length = self._sentence_lengths
        ) #a length-500 list of [num_docs, lstm_size]
        
        lstm_outputs = tf.unstack(
            tf.transpose(lstm_outputs, perm=[1, 0, 2])
        )
        
        lstm_outputs = tf.concat(
            lstm_outputs,
            axis = 0
        ) # [num_docs + MAX_SENT_LENGTH, lstm_size]
        
        mask = tf.sequence_mask(
            lengths=self._sentence_lengths,
            maxlen=MAX_DOC_LENGTH,
            dtype = tf.float32
        ) # [num_docs, MAX_SENTENCE_LENGTH]
        mask = tf.concat(tf.unstack(mask, axis = 0), axis = 0)
        mask = tf.expand_dims(mask, -1)

        lstm_outputs = mask * lstm_outputs
        lstm_outputs_split = tf.split(lstm_outputs, num_or_size_splits=self._batch_size)
        lstm_outputs_sum = tf.reduce_sum(lstm_outputs_split, axis = 1) #[num_docs, lstm_size]
        lstm_outputs_average = lstm_outputs_sum/tf.expand_dims(
            tf.cast(self._sentence_lengths, tf.float32),
            #expand dims only works with tensor of float type
            -1 # [num_docs, lstm_size]
        )

        return lstm_outputs_average


    def build_graph(self, NUM_CLASSES=None):
        embeddings = self.embedding_layer(self._data)
        lstm_outputs = self.LSTM_layer(embeddings)

        weights = tf.get_variable(
            name = 'final_layer_weights',
            shape = (self._lstm_size, NUM_CLASSES),
            initializer = tf.random_normal_initializer(seed = 2020)
        )

        biases = tf.get_variable(
            name = 'final_layer_biases',
            shape = (NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2020)
        )

        logits = tf.matmul(lstm_outputs, weights) + biases
        
        labels_one_hot = tf.one_hot(
            indices = self._labels,
            depth = NUM_CLASSES,
            dtype = tf.float32
        )
        
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels = labels_one_hot,
            logits = logits
        )
        loss = tf.reduce_mean(loss)
        
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis = 1)
        predicted_labels = tf.squeeze(predicted_labels)


    def trainer(self, loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op
