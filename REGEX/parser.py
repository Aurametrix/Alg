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

# to keep track of where we are up to in the sentence

""" 
    an index into the words array; into the list of tokens
    a stack, to which words are pushed, before popped out once their head is set
             contains words that occurred before i, for which we’re yet to assign a head.
"""

SHIFT = 0; RIGHT = 1; LEFT = 2
MOVES = [SHIFT, RIGHT, LEFT]
 
def transition(move, i, stack, parse):
    global SHIFT, RIGHT, LEFT
    if move == SHIFT:
        stack.append(i)
        return i + 1
    elif move == RIGHT:
        parse.add_arc(stack[-2], stack.pop())
        return i
    elif move == LEFT:
        parse.add_arc(i, stack.pop())
        return i
    raise GrammarError("Unknown move: %d" % move)

