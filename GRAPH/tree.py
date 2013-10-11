# tree can be represented with lists of objects
T = [["a", "b"], ["c"], ["d", ["e", "f"]]] 

class Tree: 
 def __init__(self, left, right): 
   self.left = left 
   self.right = right 


t = Tree(Tree("a", "b"), Tree("c", "d")) 
t.right.left