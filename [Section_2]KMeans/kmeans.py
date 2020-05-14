from collections import defaultdict
import numpy as np
import math
from utils import *

class KMeans:
    """
    for implementation KMeans algorithms

    ______ Attribute:
    _num_clusters: num of clusters you want to start
    _clusters: list of Clusters
    _E: list of centroids - r_d vectors
    _S: overall similarity between Members and corresponding Clusters
    _data: list of Members
    _label_count: default dict - label: num of doc have that label
    _iteration: num of iteration runed

    ______ Methods:
    load_data(self, data_path)
    random_init(self, seed_value)
    compute_similarity(self, member, centroid)
    select_cluster_for(self, member)
    update_centroid_of(self, cluster)
    stopping_condition(self, criterion, threshold)
    compute_purity(self)
    compute_NMI(self)
    run(self, seed_value, criterion, threshold)
    """
    def __init__(self, num_cluster):
        """
        init Kmeans with (num_cluster) Clusters
            (without centroids, and Members)
        :param num_cluster: num of cluster that you wanna start
        """
        self._num_cluster = num_cluster
        self._clusters = [Cluster() for _ in range(num_cluster)]
        self._E = []
        self._S = 0
        self._data = []


    def load_data(self, data_path, word_idfs_path):
        """
        load data_tfidfs and words_idfs files
            in order to get model's data and label_count
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

            #print(r_d)
            return np.array(r_d)

        #open full_precessed_data and words_idfs files
        with open(data_path) as f:
            d_lines = f.read().splitlines() #each row, each tfidf-data

        with open(word_idfs_path, 'rb') as f:
            vocab_size = len(f.read().splitlines())
        self._dims = vocab_size

        self._data = []
        self._label_count = defaultdict(int)

        #get label, doc_id, r_d vector from data_tfdifs files,
        #and put into self._data
        for d in d_lines:
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] +=1
            r_d = sparse_to_dense(features[2], vocab_size)
            self._data.append(Member(r_d, label, doc_id))

        print('Load data - Done!')

    def random_init(self, seed_value):
        """
        random init centroid of each clusters
        Choose centroids randomly from data points.
        :param seed_value: random value to seed
        :return: None
        """
        d = 0
        for cluster in self._clusters:
            d += 1
            np.random.seed(int(seed_value) + d)
            tmp = np.random.randint(len(self._data))
            cluster._centroid = self._data[tmp]._r_d


    def compute_similarity(self, member, centroid):
        """
        this func use  distance to compute similarity
            between member and centroid
        :param member: a Member object
        :param centroid: a r_d vector of Centroid
        :return: similarity = 1/Euclide_distance
        """
        mem_r_d = np.array(member._r_d)
        cen_r_d = np.array(centroid)
        dis = np.linalg.norm(mem_r_d - cen_r_d) + 1e-12

        return 1./dis

    def select_cluster_for(self, member):
        """
        select cluster - the most similar for member
        :return: max similarity of member and best fit cluster
        """
        best_fit_cluster = None
        max_simimlarity = -1

        #find the best_fit cluster in self._clusters
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_simimlarity:
                best_fit_cluster = cluster
                max_simimlarity = similarity

        #add member to best fit cluster
        best_fit_cluster.add_member(member)

        return max_simimlarity

    def update_centroid_of(self, cluster):
        """
        update centroid of cluster,
        :return None
        """
        member_r_ds = np.array([member._r_d for member in cluster._members])

        #get mean of member_r_ds @@
        avg_rd = np.mean(member_r_ds, axis = 0)

        norm2_of_avg = np.linalg.norm(avg_rd)

        #nen centroid = avg_of_mem_rd/norm2_itself **
        new_centroid = avg_rd/norm2_of_avg

        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        """
        check the stopping_condition
        + max_iters: iters over threshold
        + centroid: centroids have changed so little
        + similarity: avg_similarity has increased so little

        :return: True/ False
        """
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            #num of iterations pass over threshold max_iters
            if self._iteration >= threshold:
                return True
            else:
                return False

        elif criterion == 'centroid':
            #centroids have changed so little
            E_new = [cluster._centroid() for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]

            self._E = E_new #update _E

            #and check
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False

        else:
            #minus to check
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S #update _S

            #and check:
            if new_S_minus_S <=threshold:
                return True
            else:
                return False

    def compute_purity(self):
        """
        compute purity in order to valid clustering quality

        :return: purity of clustering
            = 1/len(data) * \
            sigma_sum (max time that a label appear in a cluster)
        """
        majoriry_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members] #list of label in cluster
            max_count = max([member_labels.count(label) for label in range(20)]) #label that appear max times in cluster
            majoriry_sum += max_count

        return majoriry_sum *1./len(self._data)

    def compute_NMI(self):
        """
        compute NMI (normalized mutual information)

        :return: NMI value of clustering
        """
        #init values:
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)

        #compute H_omega and I_value
        for cluster in self._clusters:
            wk = len(cluster._members)*1 #num of docs in cluster k
            H_omega += -wk/N * np.log10(wk/N)

            member_labels = [member._label for member in cluster._members] #list of label in cluster k
            for label in range(20):
                wk_cj = member_labels.count(label) *1 #num of mems in cluster k that have label j
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)

        #compute H_C
        for label in range(20):
            cj = self._label_count[label] * 1
            H_C += -cj/N * np.log10(cj/N)

        return I_value*2/(H_omega + H_C)

    def run(self, seed_value, criterion, threshold):
        """
        run KMeans
        :param seed_value: seed value to init clusters vs their centroids
        :param criterion: criterion for stopping
        :return: None
        """
        #init centroids:
        self.random_init(seed_value)

        ######--Print init result!
        print("Init done!")
        #####---

        #continually update clusters util convergence
        self._iteration = 0

        while True:
            print("--->iter: ", self._iteration, end = ' ')
            #reset clusters, retain only centroids
            for cluster in self._clusters:
                cluster.reset_members()

            #select cluster for members in _data
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s

            #update centroids for clusters
            for cluster in self._clusters:
                self.update_centroid_of(cluster)

            #update num of iterations and check stopping condition
            self._iteration +=1
            if self.stopping_condition(criterion, threshold):
                break
            print('done')
        print("<3 <3 Done!!! <3 <3")
