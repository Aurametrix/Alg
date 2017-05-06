Python: Algorithms, Learning resources, Modules, Updates
===

[compile to another version](https://github.com/nvbn/py-backwards)
    ➜ py-backwards -i input.py -o output.py -t 2.7
    

### Algorithms

[Sort:](SORT/)

[Regex:](REGEX/)
Examples of regular expressions for matching patterns

[Graph:](GRAPH/)
A*, B*, Breadth-first, D*, Depth-first, Dijkstra's,..

Example of a graph with six nodes A-F and eight edges represented by Python dictionary:

graph = {'A': ['B', 'C'],

'B': ['C', 'D'],

'C': ['D'],

'D': ['C'],

'E': ['F'],

'F': ['C']}          

[Machine Learning:](ML/)

[Web:](WEB/)
Beautiful Soup is a Python library for pulling data out of HTML and XML files. 

easy_install beautifulsoup4

or better keep it in virtualenv:
sudo easy_install virtualenv
pip install BeautifulSoup4

### Learning resources
[Scipy Lecture Notes](http://www.scipy-lectures.org/) 
[Python for Economics](http://quant-econ.net/py/index.html)
[Quantitative Statistics](http://people.duke.edu/~ccc14/sta-663/)

[CS109 Data Science](http://cs109.github.io/2015/pages/videos.html)

https://matterhorn.dce.harvard.edu/engage/ui/index.html#/2016/01/14328 - Harvard data science

[PYML - Machine Learning in Python](http://pyml.sourceforge.net/)

[Neural network with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

http://deeplearning.net/software/theano/ - Theano library

https://www.cs.cmu.edu/~ymiao/pdnntk.html - PDNN: A Python Toolkit for Deep Learning

http://www.southampton.ac.uk/~fangohr/training/python/pdfs/Python-for-Computational-Science-and-Engineering.pdf - Intro to Python for CS & Eng

http://rosalind.info/problems/locations/ - Platform for learning bioinformatics

http://sourceforge.net/projects/numpy/  - Numerical Python, fast and sophisticated arrays

[Pythonic Data Structures and Algorithms](https://github.com/keon/algorithms)

git clone git://github.com/numpy/numpy.git numpy

http://matplotlib.org/ - python 2D plotting library

If you get an ImportError: No module named matplotlib

set your PYTHONPATH, eg: export PYTHONPATH=/Library/Python/2.7/site-packages:$PYTHONPATH

git clone git://github.com/scipy/scipy.git scipy

PyAudio provides Python bindings for PortAudio, the cross-platform audio I/O library 

to play and record audio on a variety of platforms.

http://people.csail.mit.edu/hubert/pyaudio/

#### Simple way to access google api for speech recognition with python
https://pypi.python.org/pypi/pygsr
pip install pygsr

### security
[Gnupg](https://bitbucket.org/vinay.sajip/python-gnupg/) - Python library which takes care of the internal details and allows its users to generate and manage keys, encrypt and decrypt data, and sign and verify messages. See also
[github mirror](https://github.com/isislovecruft/python-gnupg)

### administration
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
