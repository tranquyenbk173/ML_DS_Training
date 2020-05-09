from utils import *

if __name__=="__main__":
    #kmeans_ <<20 clusters>>
    #the result was write to file with from of each line:
    #               doc_id <fff> doc_label <fff> in cluster
    clustering_vs_KMeans()

    #classifying with SMV, linear SVM and kernel SVM
    classifying_vs_SVMs()
        #result:
        #   Accuracy vs Linear SVM (C = 10, tol = 0.001):  0.8266064790228359
        #   Accuracy vs Kernel SVM (C = 50, tol = 0.001, gamma = 0.1, kernel = 'rbf'):  0.8250132766861391
