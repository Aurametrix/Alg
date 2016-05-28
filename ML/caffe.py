import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

#Set the computation mode CPU

caffe.set_mode_cpu()

#or GPU
#caffe.set_device(0)
#caffe.set_mode_gpu()

#The output map for a convolution given receptive field size has a dimension given by the following equation :

output = (input - kernel_size) / stride + 1
#Create a first file conv.prototxt describing the neuron network :

name: "convolution"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 100
input_dim: 100
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    num_output: 3
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
