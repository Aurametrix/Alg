from pypeln import pr
import time
from random import random

def slow_add1(x):
    time.sleep(random()) # <= some slow computation
    return x + 1

def slow_gt3(x):
    time.sleep(random()) # <= some slow computation
    return x > 3

data = range(10) # [0, 1, 2, ..., 9] 

stage = pr.map(slow_add1, data, workers = 3, maxsize = 4)
stage = pr.filter(slow_gt3, stage, workers = 2)

data = list(stage) # e.g. [5, 6, 9, 4, 8, 10, 7]
