if __name__ == "__main__":

    num_cluster = 20 #you can fix this
    print("With ", num_cluster, " cluster: ")
    model = kmeans.KMeans(num_cluster)
    model.load_data(data_path='./data/data_tf_idfs.txt', word_idfs_path='./data/word_idfs.txt')
