import numpy as np
import random

class DataReader:
    """
    Load data from file and store in a DataReader Object
    """
    def __init__(self, data_path, batch_size, vocab_size):
        """
        load data from data_path
        and divide into batchs with size = batch_size
        :param data_path: links to preprocessed data
        :param batch_size:
        :param vocab_size: num of words in dictionary
        """
        self._batch_size = batch_size
        self._data = [] #each member is a tfidf dense vector
        self._labels = [] #each mem is a corresponding label
        self._sentence_length = []
        self._num_epoch = 0 #init value of num of epochs
        self._batch_id = 0 #init

        #read data
        #and get label, tfidf vector from each line of data file
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>')
            label, doc_id, sentence_length, tokens  = int(features[0]), int(features[1]), int(features[2]), features[3]
            tokens = features[3].split()
            tokens = [int(token) for token in tokens]

            self._data.append(tokens)
            self._labels.append(label)
            self._sentence_length.append(sentence_length)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sentence_length = np.array(self._sentence_length)


    def next_batch(self):
        """
        get the next batch, according to current batch_id
        :return: batch of data list and labels list
        """
        #generally find the start and end indexs of data
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id +=1
        #print("batch_size = ", end - start)

        #exception, at the end part of data
        if end > len(self._data):
            end = len(self._data)
            start =  end - self._batch_size #ensure batch size

            self._num_epoch +=1 #increase num of epoch
            self._batch_id = 0 #reset batch_id
            #and shuffle data
            indices = list(range(len(self._data)))
            random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]

        return self._data[start:end], self._labels[start:end], self._sentence_length[start:end]
