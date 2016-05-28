import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

#Set the computation mode CPU

caffe.set_mode_cpu()

#or GPU
#caffe.set_device(0)
#caffe.set_mode_gpu()
