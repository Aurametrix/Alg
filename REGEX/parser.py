""" Greedy transition-based parsing
    Takes a list of string tokens as input
    Outputs a list of head indices, representing directed edges in the graph of word-word relationships
    Every node (word) will have exactly one incoming arc (one dependency, with its head), except one.
"""

class Parse(object):
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n-1)
        self.lefts = []
        self.rights = []
        for i in range(n+1):
            self.lefts.append(DefaultList(0))
            self.rights.append(DefaultList(0))
 
    def add_arc(self, head, child):
        self.heads[child] = head
        if child < head:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)
