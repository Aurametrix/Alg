class Node(object):
    "A decision-tree node."
    def evaluate(self, env):
        "My value when variables are set according to env, a dictionary."
        abstract
    def __call__(self, if0, if1):
        """Return an expression whose value is if0's or if1's,
        according as self's is 0 or 1."""
        return Choice(self, if0, if1)

class ConstantNode(Node):
    def __init__(self, value): self.value = value
    def __repr__(self):        return repr(self.value)
    def evaluate(self, env):   return self.value

class VariableNode(Node):
    def __init__(self, name):  self.name = name
    def __repr__(self):        return self.name
    def evaluate(self, env):   return env[self]

class ChoiceNode(Node):
    def __init__(self, index, if0, if1):
        self.index, self.if0, self.if1 = index, if0, if1
    def __repr__(self):
        return '%r(%r, %r)' % (self.index, self.if0, self.if1)
    def evaluate(self, env):
        branch = (self.if0, self.if1)[self.index.evaluate(env)]
        return branch.evaluate(env)
