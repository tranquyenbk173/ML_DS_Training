from utils import gather_20newsgroups_data, create_vocabulary, get_tf_idf

class pre_process:

    def __init__(self):
        return

    def bingo(self):
        gather_20newsgroups_data('data/20news-bydate.tar.gz')
        create_vocabulary('data/full_processed_data.txt')
        get_tf_idf('data/full_processed_data.txt', 'data/word_idfs.txt')