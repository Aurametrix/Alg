Python: Algorithms, Learning resources, Modules, Updates
===


###Algorithms

Sort:
Quicksort, Bubblesort, Insertionsort, Mergesort

Regex:
Examples of regular expressions for matching patterns

Graph:
Example of a graph with nodes A-F and 8 edges represented by Python dictionary:
   graph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}

Graph search algorithms:             
A*, B*, Breadth-first, D*, Depth-first, Dijkstra's,..

WEB
Beautiful Soup is a Python library for pulling data out of HTML and XML files. 

easy_install beautifulsoup4

or better keep it in virtualenv:
sudo easy_install virtualenv
pip install BeautifulSoup4

###Learning resources
http://www.scipy-lectures.org/ Scipy Lecture Notes
http://quant-econ.net/py/index.html
http://people.duke.edu/~ccc14/sta-663/

http://pyml.sourceforge.net/ - machine learning

http://deeplearning.net/software/theano/ - Theano library

https://www.cs.cmu.edu/~ymiao/pdnntk.html - PDNN: A Python Toolkit for Deep Learning

http://www.southampton.ac.uk/~fangohr/training/python/pdfs/Python-for-Computational-Science-and-Engineering.pdf - Intro to Python for CS & Eng

http://rosalind.info/problems/locations/ - Platform for learning bioinformatics


http://sourceforge.net/projects/numpy/  - Numerical Python, fast and sophisticated arrays

git clone git://github.com/numpy/numpy.git numpy

http://matplotlib.org/ - python 2D plotting library

If you get an ImportError: No module named matplotlib

set your PYTHONPATH, eg: export PYTHONPATH=/Library/Python/2.7/site-packages:$PYTHONPATH

git clone git://github.com/scipy/scipy.git scipy

PyAudio provides Python bindings for PortAudio, the cross-platform audio I/O library 

to play and record audio on a variety of platforms.

http://people.csail.mit.edu/hubert/pyaudio/

####Simple way to access google api for speech recognition with python
https://pypi.python.org/pypi/pygsr
pip install pygsr

###administration
updating 2.7.x on mac  (https://www.python.org/downloads/)

sudo rm -R /System/Library/Frameworks/Python.framework/Versions/2.7

sudo mv /Library/Frameworks/Python.framework/Versions/2.7 /System/Library/Frameworks/Python.framework/Versions

sudo chown -R root:wheel /System/Library/Frameworks/Python.framework/Versions/2.7

sudo rm /System/Library/Frameworks/Python.framework/Versions/Current

sudo ln -s /System/Library/Frameworks/Python.framework/Versions/2.7 /System/Library/Frameworks/Python.framework/Versions/Current

sudo rm /usr/bin/pydoc

sudo rm /usr/bin/python

sudo rm /usr/bin/pythonw

sudo rm /usr/bin/python-config

sudo ln -s /System/Library/Frameworks/Python.framework/Versions/2.7/bin/pydoc /usr/bin/pydoc

sudo ln -s /System/Library/Frameworks/Python.framework/Versions/2.7/bin/python /usr/bin/python

sudo ln -s /System/Library/Frameworks/Python.framework/Versions/2.7/bin/pythonw /usr/bin/pythonw

sudo ln -s /System/Library/Frameworks/Python.framework/Versions/2.7/bin/python-config /usr/bin/python-config

---

sudo easy_install pip  // pip install --upgrade pip

sudo easy_install -U numpy

pip install scipy

pip install matplotlib
