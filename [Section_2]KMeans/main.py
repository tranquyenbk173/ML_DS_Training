import kmeans

if __name__ == "__main__":
    model = kmeans.KMeans(num_cluster=20)
    model.load_data('./data/data_tf_idfs.txt', './data/word_idfs.txt')
    model.run(seed_value=13, criterion='max_iters', threshold=25)