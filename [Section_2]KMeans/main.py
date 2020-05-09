import kmeans
import numpy as np
if __name__ == "__main__":

    num_cluster = 20 #you can fix this
    print("With ", num_cluster, " cluster: ")
    model = kmeans.KMeans(num_cluster)
    model.load_data(data_path='./data/data_tf_idfs.txt', word_idfs_path='./data/word_idfs.txt')


    # Initialize centroids several times
    # and choose the one with highest NMI (or purity).
    max_NMI = -1
    purity = -1
    best_seed = 17
    centroids = []

    print("Try to get the best seed: ")
    for time in range(5):
        np.random.seed(17 + time)
        seed_value = np.random.randint(10)
        print("**Seed = ", seed_value)
        model.run(seed_value=seed_value, criterion='max_iters', threshold=25)
        NMI_value = model.compute_NMI()

        if NMI_value > max_NMI:
            max_NMI = NMI_value
            purity = model.compute_purity()
            best_seed = seed_value

    #evaluate clustering:
    print("Result: \nSeed = ", best_seed)
    print('Purity: ', purity)
    print('Best NMI: ', max_NMI)
    #___Result: 
    #   Seed =  5
    #   Purity:  0.5266369521383848
    #   Best NMI:  0.5297722233222106
