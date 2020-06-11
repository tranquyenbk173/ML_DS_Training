import re
from collections import defaultdict
from os import listdir
from os.path import isfile
from DataReader import *
from RNN import *


def gen_data_and_vocab(data_raw_path, processed_path):
    """
    create vocab: collect words in corpus,
                    and remove words that appear less frequently
    :param news_group_list:
    :return:
    """
    def collect_data_from(parent_path, newsgroup_list, word_count = None):
        """
        create vocab from data ;]]
        :param parent_path:
        :param newsgroup_list:
        :param word_count:
        :return:
        """
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + '/' + newsgroup + '/'

            files = [(filename, dir_path + filename)
                     for filename in listdir(dir_path) if isfile(dir_path + filename)]
            files.sort()
            label = group_id
            print("Processing: {}-{}".format(group_id, newsgroup))

            for filename, filepath in files:
                with open(filepath, encoding='utf-8', errors='ignore') as f:
                    text = f.read().lower() 
                    words = re.split('\W+', text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1

                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)

        return data

    word_count = defaultdict(int)

    path = data_raw_path
    parts = [path + dir_name + '/' for dir_name in listdir(path)
             if not isfile(path + dir_name)]

    train_path, test_path = (parts[0], parts[1])\
        if 'train' in parts[0] else (parts[1], parts[0])

    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]
    newsgroup_list.sort()

    #Create vocab from train data
    train_data = collect_data_from(
        parent_path = train_path,
        newsgroup_list = newsgroup_list,
        word_count=word_count
    )
        #Remove words that appear less frequently
    vocab = [word for word, freq in zip(word_count.keys(), word_count.values()) if freq > 10]
    vocab.sort()

        #and write raw vocab to file
    with open(processed_path + 'vocab-raw.txt', 'w') as f:
        f.write('\n'.join(vocab))

    #Collect data of test set
    test_data = collect_data_from(
        parent_path = test_path,
        newsgroup_list=newsgroup_list
    )

    #write to file raw train and test set
    with open(processed_path + '20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))

    with open(processed_path + '20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))


def encode_data(data_path, vocab_path, MAX_DOC_LENGTH=500, unknown_ID=0, padding_ID=1):
    """
    Assign IDs for words in vocab: 2, 3, ... V+2
    Take 2 special IDs for: unknown words and padding words
    (each vocab's encoded their words and replace them by IDs)
    :param data_path: train/ test data
    :param vocab_path: vocab that we collected
    :param MAX_DOC_LENGTH:
    :return:
    """

    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2)
                      for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])
                     for line in f.read().splitlines()]

    encoded_data = []
    
    #get info of each doc + and encode them
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH] #limit num of words in doc by MAX_DOC_Length
        sentence_length = len(words) #so <= max_doc_length??

        #Replace each word by ID
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(unknown_ID)

        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))

        encoded_data.append(str(label) + "<fff>" + str(doc_id) + "<fff>" +
                            str(sentence_length) + "<fff>" + " ".join(str(e) for e in encoded_text))
    
    #Write encoded_data to file:
    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + \
                '-encoded.txt'
    with open(dir_name + '/' + file_name, 'w') as f:
        f.write('\n'.join(encoded_data))



def get_best_hyper_param(vocab_path, train_path, list_values):
    """
    choose hyper params for model
    Grid Search with each pair (lstm_size, batch_size)
        lstm_sizes = [32, 50, 64, 64, 128]
        batch_sizes = [64, 50, 64, 128, 256]
    :param vocab_path: vocab_raw
    :param train_path: train_raw
    :return: best_param (include: lstm_size and batch_size)
    """

    def cross_validation(num_folds, initial_values):
        """

        :param num_folds:
        :param initial_values: [lstm_size, batch_size]
        :return: avg of loss
        """

        #Init hyper prams
        lstm_size, batch_size = initial_values[0], initial_values[1]

        #get train data
        with open(vocab_path) as f:
            vocab_size = len(f.read().splitlines())

        train_data_reader = DataReader(
            data_path=train_path,
            batch_size=batch_size,
            vocab_size = vocab_size
        )

        X_train, y_train, sentence_lenght_train = train_data_reader._data, train_data_reader._labels, train_data_reader._sentence_length

        #divide training set into num_folds parts follow indexes:
        row_ids = np.array(range(X_train.shape[0]))
        valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
        valid_ids[-1] = np.append(valid_ids[-1],
                                  row_ids[len(row_ids) - len(row_ids) % num_folds:])

        train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
        avg_error_rate = 0

        for i in range(num_folds):
            # with each i, we have corresponding train-val sets:
            # (k-1) parts for train and the rest for val
            valid_X_part, valid_y_part, valid_sentence_length = X_train[valid_ids[i]], y_train[valid_ids[i]], sentence_lenght_train[valid_ids[i]]
            train_X_part, train_y_part, train_sentence_length = X_train[train_ids[i]], y_train[train_ids[i]], sentence_lenght_train[train_ids[i]]
            Valid = [valid_X_part, valid_y_part, valid_sentence_length]
            Train = [train_X_part, train_y_part, train_sentence_length]

            # fit and compute corresponding RSS:
            avg_error_rate += train_and_evaluate_RNN_choose_param(vocab_path, Train, Valid,
                                                                  lstm_size, batch_size)

        return avg_error_rate / num_folds

    def range_scan(best_values, min_error_rate, list_values):
        """
        Use curr_values given, find the best_values from curr_values
        :param best_values:
        :param min_error_rate:
        :param curr_values:
        :return:
        """
        
        for values in list_values:
            error_rate = cross_validation(7, initial_values=values)
            if error_rate < min_error_rate:
                min_error_rate = error_rate
                best_values = values
                
        return best_values
    
    best_values, min_error_rate = range_scan(best_values = [32, 32], min_error_rate = 1000**2, list_values=list_values)

    return best_values
