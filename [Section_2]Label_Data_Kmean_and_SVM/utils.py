class Member:
    def __init__(self, r_d, label = None, doc_id = None):
        """
        construct a new cluster
        :param r_d: tf_idf vector of document d
        :param label: label of d
        :param doc_id: name of d
        """
        self.r_d = r_d
        self.label = label
        self.doc_id = doc_id