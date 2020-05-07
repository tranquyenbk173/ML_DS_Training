from collections import defaultdict
from os import listdir
import tarfile
from nltk.stem.porter import PorterStemmer
import re
import numpy as np

def gather_20newsgroups_data(tarFile):
    """
    read tar file given to collect data
    --> then write out the train, test, full_processed data
    :param tarFile: link tarFile containing data
    :return: None
    """
    #Collect list_newsgroups:

    tar = tarfile.open(tarFile)
    tar.extractall(path='data/data_bydate')

    parent_dir = listdir('data/data_bydate')
    parent_dir_link = 'data/data_bydate'
    train_dir, test_dir = (parent_dir[0], parent_dir[1]) if 'train' in parent_dir[0]\
        else (parent_dir[1], parent_dir[0])

    train_dir = str(parent_dir_link) + '/' + str(train_dir)
    test_dir = str(parent_dir_link) + '/' + str(test_dir)
    print(train_dir, '\n', test_dir)

    list_newsgroups = listdir(train_dir)
    list_newsgroups.sort()

    def collect_data_from(parent_dir, list_newsgroups):
        """
        collect data with form of each member:
            label<fff>file_name<fff>content
        :param parent_dir: train dir/test dir
        :param list_new_groups: list name of news groups
        :return: data (is a list of members)
        """
        data = []

        #load stop words:
        with open('data/stop_words.txt') as f:
            stop_words = f.read().splitlines()

        #Declare PorterStemmer object:
        stemmer = PorterStemmer()

        for group_id, group in enumerate(list_newsgroups):
            label = group_id
            group_path = parent_dir + '/' + group
            for file in listdir(group_path):
                file_path = group_path + '/' + file
                with open(file_path) as f:
                    text = f.read().lower()

                    #remove stop words and stem
                    words = [word for word in re.split('\W+', text)\
                             if word not in stop_words]
                    words = [stemmer.stem(word) for word in words]

                    #combine remaining words:
                    file_content = ' '.join(words)

                    data.append(str(label) + '<fff>' + str(file) + '<fff>' + str(file_content))

        return data

    #Collect train, test, full processed data
    train_processed_data = collect_data_from(train_dir, list_newsgroups)
    test_processed_data = collect_data_from(test_dir, list_newsgroups)
    full_processed_data = train_processed_data + test_processed_data

    #and write to file: train, test, full_processed_data.txt
    with open('data/train_processed_data.txt', 'w') as f:
        f.write('\n'.join(train_processed_data))
    with open('data/test_processed_data.txt', 'w') as f:
        f.write('\n'.join(test_processed_data))
    with open('data/full_processed_data.txt', 'w') as f:
        f.write('\n'.join(full_processed_data))

    print("Gather_data Done!!!")

def create_vocabulary(full_data_path):
    """
    create Vocabulary and pre-compute idfs of words in V
    write out the idfs data: with form of each line - word<fff>idf
    :param full_data_path: link to full processed data
    :return: None
    """

    def compute_idf(doc_freq, corpus_size):
        """
        compute idf(word, corpus)
        :param doc_freq: num of doc that word appear
        :param corpus_size: size of corpus
        :return: idf(word, corpus)
        """
        return np.log10(corpus_size*1./doc_freq)

    #compute corpus size
    with open(full_data_path) as f:
        lines = f.read().splitlines()

    corpus_size = len(lines)

    #compute doc_count = {(word, doc_freq of word)}
    doc_count = defaultdict(int)

    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = set(text.split())
        for word in words:
            doc_count[word] +=1

    #compute word_idfs:
    word_idfs = [(word, compute_idf(doc_freq, corpus_size))\
                 for word, doc_freq in zip(doc_count.keys(), doc_count.values())\
                 if doc_freq > 10 and not word.isdigit()]
    word_idfs.sort(key = lambda x: -x[1])

    print('Then, vocabulary size is: ', len(word_idfs))

    #write to file: word<fff>idf_of_word for each line
    with open('data/word_idfs.txt', 'w') as f:
        f.write('\n'.join([str(word) + '<fff>' + str(idf) for word, idf in word_idfs]))


def get_tf_idf(full_propressed_data_file, idfs_file):
    """
    compute tf_idf of words in each document of full_data_file
    then write to file: with form: label<fff>doc_id<fff>sparse_rep
    :param full_propressed_data_file:
    :param idfs_file:
    :return: None
    """
    #get data from pre_computed idfs and pre-processed full data:
    with open(idfs_file) as f:
        #with each line: word<fff>idf_of_word
        word_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]

    word_idfs = dict(word_idfs)
    word_IDs =  dict([(word_idfs, index) for index, word_idfs in enumerate(word_idfs)])
    word = word_idfs.keys()

    with open(full_propressed_data_file) as f:
        #with each line: label<fff>file_name<>content
        docs = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])\
                for line in f.read().splitlines()]

    #compute data_tf_idf:
    data_tf_idfs = []
    for doc in docs:
        label, id, text = doc
        #get set of words in doc:
        words = [word for word in text.split() if word in word_idfs]
        words_set = list(set(words))

        #and find max_freq in doc:
        max_freq = max(words.count(word) for word in words_set)

        #compute doc_tf_idfs of words in doc:
        doc_tf_idfs = []
        sum_squares = 0.0

        for word in words_set:
            word_freq = words.count(word)
            word_tf_idf = word_freq*1./max_freq * word_idfs[word]
            token = (word_IDs[word], word_tf_idf)
            doc_tf_idfs.append(token)
            sum_squares += word_tf_idf**2

        doc_tf_idfs_normalize = [str(index) + ':' + str(word_tf_idf/np.sqrt(sum_squares))\
                                 for index, word_tf_idf in doc_tf_idfs]

        sparse_rep = ' '.join(doc_tf_idfs_normalize)
        data_tf_idfs.append(label + '<fff>' + id + '<fff>' + sparse_rep)

    #then write data_tf_idfs to file:
    with open('data/data_tf_idfs.txt', 'w') as f:
        f.write('\n'.join(data_tf_idfs))

    print("Get TF_IDF Done!!!")


