from utils import gather_20newsgroups_data, create_vocabulary, get_tf_idf

if __name__ == "__main__":

    #gather_20newsgroups_data('data/20news-bydate.tar.gz')

    #with full data
    #create_vocabulary('data/full_processed_data.txt')
    #get_tf_idf('data/full_processed_data.txt', 'data/full_word_idfs.txt')

    #with train data
    #get_tf_idf('data/train_processed_data.txt', 'data/full_word_idfs.txt')

    #with test data
    get_tf_idf('data/test_processed_data.txt', 'data/full_word_idfs.txt')
