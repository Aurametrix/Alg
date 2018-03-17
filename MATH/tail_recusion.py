def factorial(n):
  if n == 0: return 1
  else: return factorial(n-1) * n

def tail_factorial(n, accumulator=1):
  if n == 0: return 1
  else: return tail_factorial(n-1, accumulator * n)
  


from tail_recursion import tail_recursive, recurse

# Normal recursion depth maxes out at 980, this one works indefinitely
@tail_recursive
def factorial(n, accumulator=1):
    if n == 0:
        return accumulator
    recurse(n-1, accumulator=accumulator*n)

cass Recurse(Exception):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

def recurse(*args, **kwargs):
    raise Recurse(*args, **kwargs)
        
def tail_recursive(f):
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except Recurse as r:
                args = r.args
                kwargs = r.kwargs
                continue
    return decorated
