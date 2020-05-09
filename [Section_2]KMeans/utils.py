class Cluster:
    def __init__(self):
        self._centroid = None #centroid of this cluster - r_d vector
        self._members = [] #list of Members of thís cluster

    def reset_members(self):
        #reset clusters, retain only centroids
        self._members = []

    def add_member(self, member):
        #add member to members list of this cluster
        self._members.append(member)


class Member:
    def __init__(self, r_d, label = None, doc_id = None):
        #construct a new Member vs doc's r_d vector
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id
