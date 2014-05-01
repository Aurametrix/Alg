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
             contains words that occurred before i, for which yet need to assign a head.
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

# parsing loop starts with an empty stack, and a buffer index at 0, with no dependencies recorded. 
# chooses one of the (valid) actions, and applies it to the state. Continues until the stack is empty 
# and the buffer index is at the end of the input. 

Class Parser(object):
    ...
    def parse(self, words):
        tags = self.tagger(words)
        n = len(words)
        idx = 1
        stack = [0]
        deps = Parse(n)
        while stack or idx < n:
            features = extract_features(words, tags, idx, n, stack, deps)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n, len(stack))
            next_move = max(valid_moves, key=lambda move: scores[move])
            idx = transition(next_move, idx, stack, parse)
        return tags, parse
 
def get_valid_moves(i, n, stack_depth):
    moves = []
    if i < n:
        moves.append(SHIFT)
    if stack_depth >= 2:
        moves.append(RIGHT)
    if stack_depth >= 1:
        moves.append(LEFT)
    return moves
