class DictMap:

  def __init__(self, contents):
    self.contents = contents

  def extend(self,x,y):
    contents_ = copy.copy(self.contents)
    contents_[x] = y
    return DictMap(contents_)

  def __call__(self,x):
    return self.contents[x] 
