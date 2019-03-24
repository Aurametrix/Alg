from hypothesis import given
from hypothesis.strategies import lists, floats

@given(lists(floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_mean_is_within_reasonable_bounds(ls):
    assert min(ls) <= mean(ls) <= max(ls)
    
def mean(ls):
    return sum(ls) / len(ls)
    
# overflow
    
def mean(ls):
    return sum(l / len(ls) for l in ls)
    
# lack of precision of floating point numbers


import numpy as np

def mean(ls):
    return np.array(ls).mean()
    
    # OverflowError
    
def clamp(lo, v, hi):
    return min(hi, max(lo, v))

def mean(ls):
    return clamp(min(ls), sum(ls) / len(ls), max(ls))
