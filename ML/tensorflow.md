#### Google's open source software library for numerical computation using data flow graphs written in Python, C++ and CUDA
* [TensorFlow on Github](https://github.com/tensorflow/tensorflow)
+ [Python API](https://www.tensorflow.org/api_docs/python/)


Deep Learning with Tensor Flow

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

### Tutorials

+ [getting started with tensor flow](https://www.tensorflow.org/get_started/get_started)

+ [Tensorflow vs Teano](https://news.ycombinator.com/item?id=14575465)

+ [Object detection with open images](https://blog.algorithmia.com/deep-dive-into-object-detection-with-open-images-using-tensorflow/)

+ [Resources](http://tensorflow-world-resources.readthedocs.io/en/latest/)

### Youtube 
##### Theory:
* [How NNs work: ](https://www.youtube.com/watch?v=ILsA4nyG7I0)
* [How Convolution NNs work: ](https://www.youtube.com/watch?v=FmpDIaiMIeA)
* [How Deep NNs work: ](https://www.youtube.com/watch?v=ILsA4nyG7I0&t=5s)
* [Recurrent NNs and LSTM: ](https://www.youtube.com/watch?v=WCUNPb-5EYI)

##### Applied:
* [Deep Learning with NNs and TensorFlow: ](https://www.youtube.com/watch?v=oYbVFhK_olY)


### Developed using TensorFlow

+ [Using 3D Convolutional Neural Networks for Speaker Verification](https://github.com/astorfi/3D-convolutional-speaker-recognition)

+ [Models built with TensorFlow](https://github.com/stanfordmlgroup/tf-models) from Stanford ML group

### TensorFlow Datasets and Estimators

[part 1](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html); [part 2: feature columns](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)


### Applications

+ AI agents incorporating the OpenCog AGI framework, Google Tensorflow and other powerful tools can interact within the [SingularityNET](https://github.com/singnet/singnet)

#### Stock Market Prediction

+ [Using Multi-Layer Perceptrons ](https://nicholastsmith.wordpress.com/2016/04/20/stock-market-prediction-using-multi-layer-perceptrons-with-tensorflow/)
+ [with CNN](https://github.com/kimber-chen/Tensorflow-for-stock-prediction)
+ [on GoogleCloud](https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data)


---------------
Installation
---------------

* `Installing TensorFlow`_: Official TensorFLow installation
* `Install TensorFlow from the source`_: A comprehensive guide on how to install TensorFlow from the source using python/anaconda
* `TensorFlow Installation`_: A short TensorFlow installation guide powered by NVIDIA
* `7 SIMPLE STEPS TO INSTALL TENSORFLOW ON WINDOWS`_: A concise tutorial for installing TensorFlow on Windows

.. _Installing TensorFlow: https://www.tensorflow.org/install/
.. _Install TensorFlow from the source: https://github.com/astorfi/TensorFlow-World/tree/master/docs/tutorials/installation
.. _TensorFlow Installation: http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html
.. _7 SIMPLE STEPS TO INSTALL TENSORFLOW ON WINDOWS: http://saintlad.com/install-tensorflow-on-windows/


* `Install TensorFlow on Ubuntu`_: A comprehensive tutorial on how to install TensorFlow on Ubuntu
* `Installation of TensorFlow`_: The video covers how to setup TensorFlow
* `Installing CPU and GPU TensorFlow on Windows`_: A tutorial on TensorFlow installation for Windows
* `Installing the GPU version of TensorFlow for making use of your CUDA GPU`_: A GPU-targeted TensoFlow installation


.. _Install TensorFlow on Ubuntu: https://www.youtube.com/watch?v=_3JFEPk4qQY&t=3s
.. _Installation of TensorFlow: https://www.youtube.com/watch?v=CvspEt8kSIg
.. _Installing CPU and GPU TensorFlow on Windows: https://www.youtube.com/watch?v=r7-WPbx8VuY
.. _Installing the GPU version of TensorFlow for making use of your CUDA GPU: https://www.youtube.com/watch?v=io6Ajf5XkaM

---------------
Getting Started
---------------
* `Getting Started With TensorFlow Framework`_: This guide gets you started programming in TensorFlow
* `learning TensorFlow Deep Learning`_:A great resource to start
* `Welcome to TensorFlow World`_: A simple and concise start to TensorFLow

.. _learning TensorFlow Deep Learning: http://learningtensorflow.com/getting_started/
.. _Getting Started With TensorFlow Framework: https://www.tensorflow.org/get_started/get_started
.. _Welcome to TensorFlow World: https://github.com/astorfi/TensorFlow-World/tree/master/docs/tutorials/0-welcome

* `Gentlest Introduction to Tensorflow  <https://www.youtube.com/watch?v=dYhrCUFN0eM>`_
* `TensorFlow in 5 Minutes  <https://www.youtube.com/watch?v=2FmcHiLCwTU/>`_
* `Deep Learning with TensorFlow - Introduction to TensorFlow  <https://www.youtube.com/watch?v=MotG3XI2qSs>`_
* `TensorFlow Tutorial (Sherry Moore, Google Brain)  <https://www.youtube.com/watch?v=Ejec3ID_h0w>`_
* `Deep Learning with Neural Networks and TensorFlow Introduction  <https://www.youtube.com/watch?v=oYbVFhK_olY>`_
* `A fast with TensorFlow <https:/www.youtube.com/watch?v=Q-FF_0NAT3s>`_

--------------------------
Advancing
--------------------------

* `TensorFlow Mechanics`_: More experienced machine learning users can dig more in TensorFlow
* `Advanced TensorFlow`_: Advanced Tutorials in TensorFlow
* `We Need to Go Deeper`_: A Practical Guide to Tensorflow and Inception
* `Wide and Deep Learning - Better Together with TensorFlow`_: A tutorial by Google Research Blog

.. _TensorFlow Mechanics: https://www.tensorflow.org/get_started/mnist/mechanics
.. _Advanced TensorFlow: https://github.com/sjchoi86/advanced-tensorflow
.. _We Need to Go Deeper: https://medium.com/initialized-capital/we-need-to-go-deeper-a-practical-guide-to-tensorflow-and-inception-50e66281804f
.. _Wide and Deep Learning - Better Together with TensorFlow: https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html


* `TensorFlow DeepDive`_: More experienced machine learning users can dig more in TensorFlow
* `Go Deeper - Transfer Learning`_: TensorFlow and Deep Learning
* `Distributed TensorFlow - Design Patterns and Best Practices`_: A talk that was given at the Advanced Spark and TensorFlow Meetup
* `Distributed TensorFlow Guide`_
* `Fundamentals of TensorFlow`_
* `TensorFlow Wide and Deep - Advanced Classification the easy way`_
* `Tensorflow and deep learning - without a PhD`_: A great tutorial on TensoFLow workflow



.. _TensorFlow DeepDive: https://www.youtube.com/watch?v=T0H6zF3K1mc
.. _Go Deeper - Transfer Learning: https://www.youtube.com/watch?v=iu3MOQ-Z3b4
.. _Distributed TensorFlow - Design Patterns and Best Practices: https://www.youtube.com/watch?v=YAkdydqUE2c
.. _Distributed TensorFlow Guide: https://github.com/tmulc18/Distributed-TensorFlow-Guide
.. _Fundamentals of TensorFlow: https://www.youtube.com/watch?v=EM6SU8QVSlY
.. _TensorFlow Wide and Deep - Advanced Classification the easy way: https://www.youtube.com/watch?v=WKgNNC0VLhM
.. _Tensorflow and deep learning - without a PhD: https://www.youtube.com/watch?v=vq2nnJ4g6N0


============================
Programming with TensorFlow
============================

Writing TensorFlow code.

--------------------------------
Reading data and input pipeline
--------------------------------


The first part is always how to prepare data and how to provide the pipeline to feed it to TensorFlow.
Usually providing the input pipeline can be complicated, even more than the structure design!

* `Dataset API for TensorFlow Input Pipelines`_: A TensorFlow official documentation on *Using the Dataset API for TensorFlow Input Pipelines*
* `TesnowFlow input pipeline`_: Input pipeline provided by Stanford.
* `TensorFlow input pipeline example`_: A working example.
* `TensorFlow Data Input`_: TensorFlow Data Input: Placeholders, Protobufs & Queues
* `Reading data`_: The official documentation by the TensorFLow on how to read data
* `basics of reading a CSV file`_: A tutorial on reading a CSV file
* `Custom Data Readers`_: Official documentation on this how to define a reader.

.. _Dataset API for TensorFlow Input Pipelines: https://github.com/tensorflow/tensorflow/tree/v1.2.0-rc1/tensorflow/contrib/data
.. _TesnowFlow input pipeline: http://web.stanford.edu/class/cs20si/lectures/slides_09.pdf
.. _TensorFlow input pipeline example: http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
.. _TensorFlow Data Input: https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
.. _Reading data: https://www.tensorflow.org/programmers_guide/reading_data
.. _basics of reading a CSV file: http://learningtensorflow.com/ReadingFilesBasic/
.. _Custom Data Readers: https://www.tensorflow.org/extend/new_data_formats

* `Tensorflow tutorial on TFRecords`_: A tutorial on how to transform data into TFRecords

.. _Tensorflow tutorial on TFRecords: https://www.youtube.com/watch?v=F503abjanHA

* `An introduction to TensorFlow queuing and threading`_: A tutorial on how to understand and create queues an efficient pipelines

.. _An introduction to TensorFlow queuing and threading: http://adventuresinmachinelearning.com/introduction-tensorflow-queuing/

----------
Variables
----------

~~~~~~~~~~~~~~~~~~~~~~~~
Creation, Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

* `Variables Creation and Initialization`_: An official documentation on setting up variables
* `Introduction to TensorFlow Variables - Creation and Initialization`_: This tutorial deals with defining and initializing TensorFlow variables
* `Variables`_: An introduction to variables

.. _Variables Creation and Initialization: https://www.tensorflow.org/programmers_guide/variables
.. _Introduction to TensorFlow Variables - Creation and Initialization: http://machinelearninguru.com/deep_learning/tensorflow/basics/variables/variables.html
.. _Variables: http://learningtensorflow.com/lesson2/

~~~~~~~~~~~~~~~~~~~~~~
Saving and restoring
~~~~~~~~~~~~~~~~~~~~~~

* `Saving and Loading Variables`_: The official documentation on saving and restoring variables
* `save and restore Tensorflow models`_: A quick tutorial to save and restore Tensorflow models

.. _Saving and Loading Variables: https://www.tensorflow.org/programmers_guide/variables
.. _save and restore Tensorflow models: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

~~~~~~~~~~~~~~~~~
Sharing Variables
~~~~~~~~~~~~~~~~~

* `Sharing Variables`_: The official documentation on how to share variables

.. _Sharing Variables: https://www.tensorflow.org/programmers_guide/variable_scope

* `Deep Learning with Tensorflow - Tensors and Variables`_: A Tensorflow tutorial for introducing Tensors, Variables and Placeholders
* `Tensorflow Variables`_: A quick introduction to TensorFlow variables
* `Save and Restore in TensorFlow`_: TensorFlow Tutorial on Save and Restore variables

.. _Deep Learning with Tensorflow - Tensors and Variables: https://www.youtube.com/watch?v=zgV-WzLyrYE
.. _Tensorflow Variables: https://www.youtube.com/watch?v=UYyqNH3r4lk
.. _Save and Restore in TensorFlow: https://www.tensorflow.org/programmers_guide/variable_scope

--------------------
TensorFlow Utilities
--------------------



~~~~~~~~~~
Supervisor
~~~~~~~~~~

* `Supervisor - Training Helper for Days-Long Trainings`_: The official documentation for TensorFLow Supervisor.
* `Using TensorFlow Supervisor with TensorBoard summary groups`_: Using both TensorBoard and the Supervisor for profit
* `Tensorflow example`_: A TensorFlow example using Supervisor.


.. _Supervisor - Training Helper for Days-Long Trainings: https://www.tensorflow.org/programmers_guide/supervisor
.. _Using TensorFlow Supervisor with TensorBoard summary groups: https://dev.widemeadows.de/2017/01/21/using-tensorflows-supervisor-with-tensorboard-summary-groups/
.. _Tensorflow example: http://codata.colorado.edu/notebooks/tutorials/tensorflow_example_davis_yoshida/

~~~~~~~~~~~~~~~~~~~
TensorFlow Debugger
~~~~~~~~~~~~~~~~~~~

* `TensorFlow Debugger (tfdbg) Command-Line-Interface Tutorial`_: Official documentation for using debugger for MNIST
* `How to Use TensorFlow Debugger with tf.contrib.learn`_: A more high-level method to use the debugger.
* `Debugging TensorFlow Codes`_: A Practical Guide for Debugging TensorFlow Codes
* `Debug TensorFlow Models with tfdbg`_:  A tutorial by Google Developers Blog


.. _TensorFlow Debugger (tfdbg) Command-Line-Interface Tutorial: https://www.tensorflow.org/programmers_guide/debugger
.. _How to Use TensorFlow Debugger with tf.contrib.learn: https://www.tensorflow.org/programmers_guide/tfdbg-tflearn
.. _Debugging TensorFlow Codes: https://github.com/wookayin/tensorflow-talk-debugging
.. _Debug TensorFlow Models with tfdbg: https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html

~~~~~~~~~~
MetaGraphs
~~~~~~~~~~

* `Exporting and Importing a MetaGraph`_: Official TensorFlow documentation
* `Model checkpointing using meta-graphs in TensorFlow`_: A working example

.. _Exporting and Importing a MetaGraph: https://www.tensorflow.org/programmers_guide/meta_graph
.. _Model checkpointing using meta-graphs in TensorFlow: http://www.seaandsailor.com/tensorflow-checkpointing.html

~~~~~~~~~~~
Tensorboard
~~~~~~~~~~~

* `TensorBoard - Visualizing Learning`_: Official documentation by TensorFlow.
* `TensorFlow Ops`_: Provided by Stanford
* `Visualisation with TensorBoard`_: A tutorial on how to create and visualize a graph using TensorBoard
* `Tensorboard`_: A brief tutorial on Tensorboard

.. _TensorBoard - Visualizing Learning: https://www.tensorflow.org/get_started/summaries_and_tensorboard
.. _TensorFlow Ops: http://web.stanford.edu/class/cs20si/lectures/notes_02.pdf
.. _Visualisation with TensorBoard: http://learningtensorflow.com/Visualisation/
.. _Tensorboard: http://edwardlib.org/tutorials/tensorboard


* `Hands-on TensorBoard (TensorFlow Dev Summit 2017)`_: An introduction to the amazing things you can do with TensorBoard
* `Tensorboard Explained in 5 Min`_: Providing the code for a simple handwritten character classifier in Python and visualizing it in Tensorboard
* `How to Use Tensorboard`_: Going through a bunch of different features in Tensorboard


.. _Hands-on TensorBoard (TensorFlow Dev Summit 2017): https://www.youtube.com/watch?v=eBbEDRsCmv4
.. _Tensorboard Explained in 5 Min: https://www.youtube.com/watch?v=3bownM3L5zM
.. _How to Use Tensorboard: https://www.youtube.com/watch?v=fBVEXKp4DIc

====================
TensorFlow Tutorials
====================

This section is dedicated to provide tutorial resources on the implementation of
different models with TensorFlow.

------------------------------
Linear and Logistic Regression
------------------------------


* `TensorFlow Linear Model Tutorial`_: Using TF.Learn API in TensorFlow to solve a binary classification problem
* `Linear Regression in Tensorflow`_: Predicting house prices in Boston area
* `Linear regression with Tensorflow`_: Make use of tensorflow for numeric computation using data flow graphs
* `Logistic Regression in Tensorflow with SMOTE`_: Implementation of Logistic Regression in TensorFlow
* `A TensorFlow Tutorial - Email Classification`_: Using a simple logistic regression classifier
* `Linear Regression using TensorFlow`_: Training a linear model by TensorFlow
* `Logistic Regression using TensorFlow`_: Training a logistic regression by TensorFlow for binary classification


.. _TensorFlow Linear Model Tutorial: https://www.tensorflow.org/tutorials/wide
.. _Linear Regression in Tensorflow: https://aqibsaeed.github.io/2016-07-07-TensorflowLR/
.. _Linear regression with Tensorflow: https://www.linkedin.com/pulse/linear-regression-tensorflow-iv%C3%A1n-corrales-solera
.. _Logistic Regression in Tensorflow with SMOTE: https://aqibsaeed.github.io/2016-08-10-logistic-regression-tf/
.. _A TensorFlow Tutorial - Email Classification: http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html
.. _Linear Regression using TensorFlow: https://github.com/astorfi/TensorFlow-World/tree/master/docs/tutorials/2-basics_in_machine_learning/linear_regression
.. _Logistic Regression using TensorFlow: https://github.com/astorfi/TensorFlow-World/tree/master/docs/tutorials/2-basics_in_machine_learning/logistic_regression

* `Deep Learning with Tensorflow - Logistic Regression`_: A tutorial on Logistic Regression
* `Deep Learning with Tensorflow - Linear Regression with TensorFlow`_: A tutorial on Linear Regression

.. _Deep Learning with Tensorflow - Logistic Regression: https://www.youtube.com/watch?v=4cBRxZavvTo&t=1s
.. _Deep Learning with Tensorflow - Linear Regression with TensorFlow: https://www.youtube.com/watch?v=zNalsMIB3NE


