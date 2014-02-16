Alg
===

Algorithms

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

http://pyml.sourceforge.net/ - machine learning

http://sourceforge.net/projects/numpy/  - Numerical Python, fast and sophisticated arrays
git clone git://github.com/numpy/numpy.git numpy

http://matplotlib.org/ - python 2D plotting library
If you get an ImportError: No module named matplotlib
set your PYTHONPATH, eg: export PYTHONPATH=/Library/Python/2.7/site-packages:$PYTHONPATH

git clone git://github.com/scipy/scipy.git scipy

PyAudio provides Python bindings for PortAudio, the cross-platform audio I/O library 
to play and record audio on a variety of platforms.
http://people.csail.mit.edu/hubert/pyaudio/

Simple way to access google api for speech recognition with python
https://pypi.python.org/pypi/pygsr
pip install pygsr
