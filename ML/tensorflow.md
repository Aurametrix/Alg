 docker run -it tensorflow/tensorflow bash
 
 apt-get update; apt-get install -y git
 git clone https://github.com/aymericdamien/TensorFlow-Examples
 
 cd TensorFlow-Examples/examples/3_NeuralNetworks
 time python convolutional_network.py
 
 nvidia-docker run --rm nvidia/cuda nvidia-smi
 nvidia-docker run --rm nvidia/cuda:7.5 nvidia-smi
 
 nvidia-docker run -it tensorflow/tensorflow:latest-gpu bash
  
 apt-get update; apt-get install -y git
 git clone https://github.com/aymericdamien/TensorFlow-Examples
 
 cd TensorFlow-Examples/examples/3_NeuralNetworks
 time python convolutional_network.py
 
  cd ../5_MultiGPU
 time python multigpu_basics.py


+ [getting started with tensor flow](https://www.tensorflow.org/get_started/get_started)

+ [Tensorflow vs Teano](https://news.ycombinator.com/item?id=14575465)

### Developed using TensorFlow

+ [Using 3D Convolutional Neural Networks for Speaker Verification](https://github.com/astorfi/3D-convolutional-speaker-recognition)
