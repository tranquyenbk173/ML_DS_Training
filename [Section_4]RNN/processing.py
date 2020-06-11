from utils import *

if __name__ == '__main__':
    gen_data_and_vocab(data_raw_path = '20news-bydate/', processed_path= 'Processed_data/')

    encode_data(data_path='Processed_data/20news-train-raw.txt',
                 vocab_path = 'Processed_data/vocab-raw.txt')

    encode_data(data_path='Processed_data/20news-test-raw.txt',
                 vocab_path = 'Processed_data/vocab-raw.txt')
