from collections import defaultdict

import numpy as np

def load_data(data_path, word_idfs_path):
    """
    load data_tfidfs and words_idfs files
        in order to get data and label
    :param data_path: link to data_tf_idfs file
    :return: None
    """
    def sparse_to_dense(sparse_r_d, vocab_size):
        """
        transform r_d into vector that dims = vocab_size
        positions that in "sparse r_d" --> # 0
            else ---> = 0
        Eg. Form of sparse_r_d
        [14174:0.008788634650700206 13720:0.030925826829923748...]
        :param sparse_r_d: "sparse r_d" from file
        :param vocab_size: size of vocab
        :return: dense r_d vector
        """
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf

        return np.array(r_d)

    #open full_precessed_data and words_idfs files
    with open(data_path) as f:
        d_lines = f.read().splitlines() #each row, each tfidf-data

    with open(word_idfs_path, 'rb') as f:
        vocab_size = len(f.read().splitlines())

    #get label, r_d vector and id of docs from data_tfdifs files
    data = []
    labels = []
    ids = []

    for d in d_lines:
        features = d.split('<fff>')
        label = int(features[0])
        labels.append(label)
        r_d = sparse_to_dense(features[2], vocab_size)
        data.append(r_d)
        id = int(features[1])
        ids.append(id)

    return data, labels, ids

def clustering_vs_KMeans():
    """
    use sklearn.KMeans to clustering text data
    save the result as num_cluster = 20 = num_labels
    :return:
    """
    data, labels, ids = load_data(data_path='data/full_data_tf_idfs.txt', word_idfs_path='data/word_idfs.txt')

    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix

    X = csr_matrix(data) #use csr_matrix to create a sparse matrix with efficient row slicing
    print('++++++++')
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5, #num of time that kmeans runs with diff initialized centroids
        tol=1e-3, #threshold for acceptable minimum error decrease
        random_state=2020 #set to get deterministic results
    ).fit(X)

    clusters = kmeans.labels_

    #write the result to file --- //observe//
    result = []
    for i in range(len(data)):
        result.append(str(ids[i]) + '<fff>' + str(labels[i]) + '<fff>' + str(clusters[i]))

    with open('data/result_sklearn_KMeans.txt', 'w') as f:
        f.write('\n'.join(result))


def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float))/len(expected_y)
    return accuracy

def classifying_vs_SVMs():
    train_X, train_y, ids = load_data(data_path='data/train_tf_idfs.txt', word_idfs_path='data/word_idfs.txt')
    test_X, test_y, ids  = load_data(data_path='data/test_tf_idfs.txt', word_idfs_path='data/word_idfs.txt')

    from sklearn.svm import LinearSVC, SVC

    ############# linear SVM #########################
    classifer1 = LinearSVC(
        C = 10.0, #penalty coeff
        tol = 0.001, #tolerance for stopping criteria
        #verbose=True #whether prints out logs or not
    )
    classifer1.fit(train_X, train_y)
    predicted_y1 = classifer1.predict(test_X)
    accuracy1 = compute_accuracy(predicted_y1, test_y)
    print("Accuracy vs Linear SVM (C = 10, tol = 0.001): ", accuracy1)

    ############## kernel SVC##########################
    classifer2 = SVC(
        C = 50.0,
        kernel = 'rbf', #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        gamma = 0.1,
        tol = 0.001,
        verbose = False
    )
    classifer2.fit(train_X, train_y)
    predicted_y2 = classifer2.predict(test_X)
    accuracy2 = compute_accuracy(predicted_y2, test_y)
    print("Accuracy vs Kernel SVM (C = 50, tol = 0.001, gamma = 0.1, kernel = 'rbf'): ", accuracy2)
