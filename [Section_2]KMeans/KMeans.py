from collections import defaultdict
import numpy as np

from utils import *

class KMeans:
    def __init__(self, initial_clusters):
        """
        Construct a new KMeans model object
        ... in order to implement KMeans algorithms
        Start with initialize 'empty' clusters
        (no member, no centroid, only num of clusters)

        :param initial_clusters:
            number of initial clusters
        """
        self.initial_clusters = initial_clusters
        self.clusters = [Cluster() for _ in \
                         range(initial_clusters)]
        self.E = [] #list of cetroids
        self.S = 0 #overall similarity

    def load_data(self, data_path, data_tf_idfs_path):
        """
        load data from tf_idfs file of cropus D
        --> into Members in self.data property
        (and 'sparse to dense' it)
        :param data_path: link to data file
        :param data_tf_idfs_path:
        :return: None
        """

        def sparse_to_dense(sparse_r_d, vocab_size):
            """
            transform r_d, from sparse to dense with size of vocabsize
            eg: vocal size = 5, sparse_rd = [2:1.2 3:1.45]
            ---> rd = [0, 0, 1.2, 1.45, 0]

            :param sparse_r_d: sparse r_d
            :param vocab_size: size of Vocal :))
            :return: dense r_d (numpy array)
            """
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidfs in indices_tfidfs:
                index = int(index_tfidfs.split(':')[0])
                tfidf = float(index_tfidfs.split(':')[1])
                r_d[index] = tfidf

            return np.array(r_d)

        #load data from data file and get vocab size from tfidfs file
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        with open(data_tf_idfs_path) as f:
            vocab_size = len(f.read().splitlines())

        #make data (as list of Member) and label_count of data
        self.data =  [] #including Members - <data points>
        self.label_count = defaultdict(int) #Ghi nhan so luong van ban \
                                            # thuoc mot nhan nao do
        for d in d_lines:
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self.label_count[label] +=1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)

            self.data.append(Member(r_d, label, doc_id))

    def random_init(self, seed_value):
        """

        :param seed_value:
        :return:
        """



    def compute_similarity(self, member, centroid):
        """

        :param member:
        :param centroid:
        :return:
        """

    def select_cluster_for(self, member):
        """

        :param member:
        :return: max_sililarity
        """
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self.clusters:
            similarity = self.compute_similarity(member, cluster.centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity

        best_fit_cluster.add_member(member)

        return max_similarity

    def update_centroid_of(self, cluster):
        """

        :param cluster:
        :return:
        """
        member_r_ds = [member.r_d for member in cluster.members]
        avg_r_d = np.mean(member_r_ds, axis = 0)
        sqrt_sum_sqr  = np.sqrt(np.sum(avg_r_d**2))
        new_centroid = np.array([value/sqrt_sum_sqr for value in avg_r_d])

        cluster.centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        """
        check stopping condition for alg vs criteria = ['centroid', 'similarity', 'max_iters']
        :param criterion:
        :param threshold:
        :return:
        """
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            if self.iteration >= threshold:
                return True
            else:
                False

        elif criterion == 'centroid':
            E_new = [list(cluster.centroid) for cluster in self.clusters]
            E_new_minus_E = [centroid for centroid in E_new\
                             if centroid not in self.E]
            self.E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False

        else:
            new_S_minus_S = self.new_S - self.S
            self.S = self.new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False

        self.new_S = 0
        for member in self.data:
            max_s = self.select_cluster_for(member)
            self.new_S += max_s


    def run(self, seed_value, criterion, threshold):
        """

        :param seed_value:
        :param criterion:
        :param threshold:
        :return:
        """
        # continually update cluster until convergence
        self.iteration = 0
        while True:
            # reset clusters, retain only centroids:
            for cluster in self.clusters:
                cluster.reset_members()

            self.new_S = 0
            for member in self.data:
                max_s = self.select_cluster_for(member)
                self.new_S += max_s

            for cluster in self.clusters:
                self.update_centroid_of(cluster)

            self.iteration += 1
            if self.stopping_condition(criterion, threshold):
                break

    def compute_purity(self):
        """

        :return:
        """
        majority_sum = 0
        for cluster in self.clusters:
            member_labels = [member.label for member in cluster.members]
            max_count = max([member_labels.count(label) for label in range (20)])
            majority_sum += max_count

        return majority_sum*1./len(self.data)

    def compute_NMI(self):
        """
        :return:
        """