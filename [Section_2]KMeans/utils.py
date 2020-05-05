class Member:
    """
    data point - member in cluster
    """
    def __init__(self, r_d, label = None, doc_id = None):
        """
        construct a new data point - member in cluster
        :param r_d: tf_idf vector of document d
        :param label: label of d
        :param doc_id: name of d
        """
        self.r_d = r_d
        self.label = label
        self.doc_id = doc_id


class Cluster:
    def __init__(self):
        """
        construct a new cluster,
        with no centroid and members
        """
        self.centroid = None
        self.members = []

    def reset_members(self):
        """
        make members set of cluster ->> NULL
        :return: None
        """
        self.members = []

    def add_member(self, member):
        """
        add member to members set
        :param member: expected member
        :return: None
        """
        self.members.append(member)


#