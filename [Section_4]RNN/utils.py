import re
from collections import defaultdict
from os import listdir
from os.path import isfile


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
    
