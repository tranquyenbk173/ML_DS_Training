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
        self._num_epoch = 0 #init value of num of epochs
        self._batch_id = 0 #init

        #read data
        #and get label, tfidf vector from each line of data file
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            vector = [0.0 for _ in range(vocab_size)] #to make dense tf-ifd vector
            for token in tokens:
                index, value = int(token.split(':')[0]), float(token.split(':')[1])
                vector[index] = value

            self._data.append(vector)
            self._labels.append(label)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)


    def next_batch(self):
        """
        get the next batch, according to current batch_id
        :return: batch of data list and labels list
        """
        #generally find the start and end indexs of data
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id +=1

        #exception, at the end part of data
        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch +=1 #increase num of epoch
            self._batch_id = 0 #reset batch_id
            #and shuffle data
            indices = list(range(len(self._data)))
            random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]

        return self._data[start:end], self._labels[start:end]


def save_parameters(name, value, epoch):
    """
    Save parameters of training process
    (At any step you want)
    :param name: name of variable
    :param value: value of variable
    :param epoch: number of saved epoch
    :return: None
    """

    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    if len(value.shape) == 1: #is a list
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number)
                                for number in value[row]])
                                for row in range(value.shape[0])])
    with open('saved_paras/' + filename, 'w') as f:
        f.write(string_form)


def restore_parameters(name, epoch):
    """
    Restore saved params
    :param name: name of variable
    :param epoch: corresponding epoch of saved variable
    :return: value of variable
    """
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    with open('saved_paras/' + filename) as f:
        lines = f.read().splitlines()

    if len(lines) == 1: #is a vector
        value = [float(number) for number in lines[0].split(',')]
    else: #is a matrix
        value = [[float(number) for number in lines[row].split(',')]
                 for row in range(len(lines))]

    return value
