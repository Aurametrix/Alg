Python
===

### Modules, Updates, Setup

[Python programming language](https://github.com/python/cpython)
[Top libraries of 2020](https://tryolabs.com/blog/2020/12/21/top-10-python-libraries-of-2020/)
[Pyston](https://blog.pyston.org/2020/10/28/pyston-v2-20-faster-python/)

[compile to another version](https://github.com/nvbn/py-backwards)
    ➜ py-backwards -i input.py -o output.py -t 2.7
    
[Nuitka, a Python compiler](http://nuitka.net/pages/overview.html)
 
[REPL, online compiler](https://repl.it/repls/FrugalOrderlyLearning)
 
[Junja2 - a template engine](http://jinja.pocoo.org/); [2.1](https://github.com/pallets/jinja/releases/tag/2.10)

[Snakemake workflow management system](http://snakemake.readthedocs.io/en/stable/) - is a tool to create reproducible and scalable data analyses

[Setting up Python for ML on Windows](https://realpython.com/python-windows-machine-learning-setup/)

[Machine Learning Tutorial](https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/)

[A Simple Guide to creating Predictive Models](https://medium.com/datadriveninvestor/a-simple-guide-to-creating-predictive-models-in-python-part-2b-7be3afb5c557)

[IDOM](https://rmorshea.github.io/articles/2021/idom-react-but-its-python/article/) - declarative Python package for building highly interactive user interfaces

[Python IDM](https://github.com/pyIDM/PyIDM) - Internet Download manager

[Features of Python3](https://github.com/arogozhnikov/python3_with_pleasure)

+ [How Python Attributes work](https://tenthousandmeters.com/blog/python-behind-the-scenes-7-how-python-attributes-work/)

+ Python libraries for NLP:
    + Natural Language Toolkit (NLTK): https://www.nltk.org/
        + [War and Peace with NLTK](http://csmoon-ml.com/index.php/2019/01/25/analysis-of-text-tolstoys-war-and-peace/)
    + spaCy: https://spacy.io/
    + TextBlob: https://github.com/sloria/TextBlob/
    + Chatterbot: https://chatterbot.readthedocs.io/en/stable/
    
    + Gensim: https://radimrehurek.com/gensim/ - for similarity analysis

[Top 10 libraries of 2019](https://tryolabs.com/blog/2019/12/10/top-10-python-libraries-of-2019/)

[Built-ins](https://treyhunner.com/2019/05/python-builtins-worth-learning/)

[Composing programs](https://composingprograms.com/)

[Free for Developers](https://free-for.dev/#/)
    + [Atlas toolkit](https://atlastk.org/) - Lightweight library to develop single-page web applications that are instantly accessible. Available for Java, Node.js, Perl, Python and Ruby.
   + [Colaboratory](https://colab.research.google.com/)  — Free web-based Python notebook environment with Nvidia Tesla K80 GPU.
   + [Datapane](https://datapane.com/)


[job queues](https://github.com/rq/rq)

[Algebraic Number Theory package](https://github.com/louisabraham/algnuth)

[Loops](https://www.blog.duomly.com/loops-in-python-comparison-and-performance/)
[F-strings](https://realpython.com/python-f-strings/)

[The Python scientific stack, compiled to WebAssembly ](https://github.com/iodide-project/pyodide); see [demo](https://iodide.io/pyodide-demo/python.html)

[Functional Programming](https://github.com/dry-python/returns)

[generations vs functions](https://www.pythonkitchen.com/python-generators-in-depth/)

[Coding habits for data scientists](https://www.thoughtworks.com/insights/blog/coding-habits-data-scientists) 

[Clean Code](https://github.com/davified/clean-code-ml/blob/master/notebooks/titanic-notebook-1.ipynb)

[Remote GUI for Python](http://www.remigui.com/)

[AI Autocomplete](https://transformer.huggingface.co/); [code ](https://github.com/huggingface/transfer-learning-conv-ai)

[Pyodide, Python in a web browser](https://alpha.iodide.io/notebooks/222/)

[online translation server](https://github.com/translate/pootle)

[NLP library](https://stanfordnlp.github.io/stanfordnlp/)
      + [Bran](https://github.com/patverga/bran) - relation extraction based purely on attention
   
[T5](https://github.com/google-research/text-to-text-transfer-transformer)

[pyGeno](http://pygeno.iric.ca/) for precision medicine -- [github](https://github.com/tariqdaouda/pyGeno)

[Library for quantitative finance](https://www.quantlib.org/)

[make a self contained executable for windows](https://pypi.org/project/py2exe/)

[spaCy](https://blog.dominodatalab.com/natural-language-in-python-using-spacy/?r=1) for Text Analytics

[Subinterpreters](https://lwn.net/SubscriberLink/820424/172e6da006687167/)

[Python Turtle](https://dev.to/ducaale/teaching-code-reuse-using-python-turtle-5fmd) for code reuse

[Pylance](https://www.infoq.com/news/2020/07/pylance-visual-studio-code/)

[Dependency management tools](https://modelpredict.com/python-dependency-management-tools)

#### Jupyter Notebooks

+ [nb2md: conversion of Jupyter and Zeppelin notebooks to Jupyter or Markdown formats](https://github.com/elehcimd/nb2md/)

+ [Jupyter Notebook to Web Apps](https://github.com/ChristianFJung/NotebookToWebApp/blob/master/article.md)


    jupyter notebook --generate-config
     
    notepad C:\Users\[user-name]\.jupyter\jupyter_notebook_config.py
    
    c.NotebookApp.notebook_dir ='C:/the/path/to/home/folder/'  # to change directory
    c.NotebookApp.notebook_dir = 'C:\\username\\folder_that_you_whant'  # for windows
    
  Go to your Jupyter Notebook link and right click it. Select properties. Go to the Shortcut menu and click Target. Look for %USERPROFILE%. Delete it. Save. Restart Jupyter.
  C:\xxx\cwp.py C:xxx\envs\E1 C:\xxx\python.exe C:\xxx\jupyter-notebook-script.py "%USERPROFILE%/"  
    
    #c.NotebookApp.token = 'nonempty-string'  # remove generated if "'_xsrf' argument missing from POST
    
    c.NotebookApp.disable_check_xsrf = True 
    

### Conferences

+ [2019 Scipy](https://www.scipy2019.scipy.org/)
    + [2017 Scipy Workshop](https://github.com/Andrewnetwork/WorkshopScipy)

+ [2019 PyCon](https://us.pycon.org/2019/)


#### Discussions

[Python news](https://news.python.sc/newest)

[New Pandas](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.0.0.html) - 1.00 - January 29, 2020

[what's new in 1.0.0](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.0.0.html)

[Is Python the world's most popular language?](https://news.ycombinator.com/item?id=18182003)


### tips & tricks

pass a URL in place of a file name 
    dfs = pd.read_html(url)

    date ranges date_range = pd.date_range(date_from, date_to, freq="D")

if you set `indicator` parameter of merge() to True pandas adds a column that tells you which dataset the row came from merge with approximate match - the tolerance parameter of merge_asof()

    pd.merge_asof(trades, quotes, on="timestamp", by='ticker', tolerance=pd.Timedelta('10ms'), direction='backward')

Merge with indicator is also useful for doing anti-joins:

    left.merge(right, how="left", indicator=True, ...) 
    [lambda df: df._merge == "left_only"]

Use gzip with when saving to csv

Create an Excel report and add some charts 

+ [write beetter python](https://github.com/SigmaQuan/Better-Python-59-Ways)

+ [API Documentation for Python Projects](https://pdoc.dev/)


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

+ [Structural Pattern Matching](https://lwn.net/Articles/838600/)
[Machine Learning:](ML/)

+ [ML Crash Course: The Bias-Variance Dilemma](https://ml.berkeley.edu/blog/2017/07/13/tutorial-4/)
+ [Reinforcement Learning](https://github.com/5vision)
+ [Chatfirst API](https://github.com/chatfirst/chatfirst)

+ [Open source deep learning models](https://github.com/samdeeplearning/The-Terrible-Deep-Learning-List)
+ [the future of deep learning](https://blog.keras.io/the-future-of-deep-learning.html)

+ [in-browser deep learning](https://tenso.rs/#readmore)

+ [Collaboratory](https://research.google.com/colaboratory/unregistered.html)

+ [Azure notebooks](https://notebooks.azure.com/)

+ Amazon's Machine Learning University (MLU) 
    + [NLP](https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp)
    + [Computer Vision](https://github.com/aws-samples/aws-machine-learning-university-accelerated-cv)
    + [Tabulr class](https://github.com/aws-samples/aws-machine-learning-university-accelerated-tab)
    
+ [CoCalc](https://cocalc.com/) created by the SageMath developers


[Web:](WEB/)
Beautiful Soup is a Python library for pulling data out of HTML and XML files. 

easy_install beautifulsoup4

or better keep it in virtualenv:
sudo easy_install virtualenv
pip install BeautifulSoup4

+ [Pelican for Web building](https://shahayush.com/2020/03/web-pelican-intro/)

+ [Data Science handbook]{(https://jakevdp.github.io/PythonDataScienceHandbook/)
+ [Statistics for Hackers](https://www.youtube.com/watch?v=Iq9DzN6mvYA)

[5 web scraping libraries](https://elitedatascience.com/python-web-scraping-libraries)
+The Farm: Requests
+The Stew: Beautiful Soup 4
+The Salad: lxml
+The Restaurant: Selenium
+The Chef: Scrapy


### Learning resources
* [Pytudes](https://github.com/norvig/pytudes) - Norvig's practice programs
* [Scipy Lecture Notes](http://www.scipy-lectures.org/) 
* [Python for Economics](http://quant-econ.net/py/index.html)
* [Quantitative Statistics](http://people.duke.edu/~ccc14/sta-663/)
* [CS topics](https://github.com/jwasham/coding-interview-university)
* [Python for Kids](https://github.com/mytechnotalent/Python-For-Kids)
* [Python in Pics](https://projects.raspberrypi.org/en/codeclub/python-module-1)
* [Intro for absolute beginners](https://github.com/webartifex/intro-to-python)
* [Projects for Beginners](https://www.codewithrepl.it/python-projects-for-beginners.html)
* [Coconut](http://coconut-lang.org/), a functional programming language that compiles to Python

* [Google Collab](https://colab.research.google.com/notebooks/welcome.ipynb#)

* [CS109 Data Science](http://cs109.github.io/2015/pages/videos.html)

* [Harvard Data Science](https://matterhorn.dce.harvard.edu/engage/ui/index.html#/2016/01/14328)

* [PYML - Machine Learning in Python](http://pyml.sourceforge.net/)

* [Neural network with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

* [Theano Library](http://deeplearning.net/software/theano/)

* (https://www.cs.cmu.edu/~ymiao/pdnntk.html) - PDNN: A Python Toolkit for Deep Learning

* [Intro to Python for CS & Eng](http://www.southampton.ac.uk/~fangohr/training/python/pdfs/Python-for-Computational-Science-and-Engineering.pdf)

* [MS 44-part programming course](https://www.youtube.com/playlist?list=PLlrxD0HtieHhS8VzuMCfQD4uJ9yne1mE6)

* [Platform for learning bioinformatics](http://rosalind.info/problems/locations/)
* [Common mistakes of beginners](https://deepsource.io/blog/python-common-mistakes/)
* [Matrix Calculus](http://www.matrixcalculus.org/)

* [API checklist](http://python.apichecklist.com/)

* [Numerical Python, fast and sophisticated arrays](http://sourceforge.net/projects/numpy/) - [python 2.7](https://github.com/numpy/numpy/blob/master/doc/neps/dropping-python2.7-proposal.rst)

* [Python cheet sheets](https://www.pythonsheets.com/)

* [Pythonic Data Structures and Algorithms](https://github.com/keon/algorithms)

    git clone git://github.com/numpy/numpy.git numpy

    http://matplotlib.org/ - python 2D plotting library

If you get an ImportError: No module named matplotlib

set your PYTHONPATH, eg: export PYTHONPATH=/Library/Python/2.7/site-packages:$PYTHONPATH

    git clone git://github.com/scipy/scipy.git scipy

PyAudio provides Python bindings for PortAudio, the cross-platform audio I/O library 

to play and record audio on a variety of platforms.

http://people.csail.mit.edu/hubert/pyaudio/

[funct array](https://github.com/Lauriat/funct) - a better python list


[Simple way to access google api for speech recognition with python](https://pypi.python.org/pypi/pygsr)
pip install pygsr

[A grammar of graphics for python](https://github.com/has2k1/plotnine)

[Knowledge extraction from unstructured texts](https://blog.heuritech.com/2016/04/15/knowledge-extraction-from-unstructured-texts/)

[wrapper providing R's ggplot2 syntax](https://github.com/sirrice/pygg)

[Brancher](https://brancher.org/), A user-centered Python package for differentiable probabilistic inference

[Automate the boring stuff](https://automatetheboringstuff.com/2e/)

[Dev environments](https://jacobian.org/2019/nov/11/python-environment-2020/)

[the Hitchhikers Guide to Python](https://docs.python-guide.org/)

[A Hitchhikers Guide to Asynchronous Programming](https://github.com/crazyguitar/pysheeet/blob/master/docs/appendix/python-concurrent.rst)

### Interesting projects
+ [Generate Hacker News Projects](https://hncynic.leod.org/), [code](https://github.com/leod/hncynic)
+ [Find usernames across social networks](https://github.com/sherlock-project/sherlock)
+ [HN API](https://github.com/HackerNews/API) = [Algolia](https://hn.algolia.com/api)
+ [Rust bindings for Python](https://github.com/PyO3/pyo3)
+ [Pen Plotter](https://github.com/evildmp/BrachioGraph)
+ [web-based spreadsheet application](https://github.com/ricklamers/gridstudio)
+ [Wonderland of math](https://github.com/neozhaoliang/pywonderland)
+ [Quantitative Finance](https://github.com/cantaro86/Financial-Models-Numerical-Methods)
+ [Simulating Quantum mechanics](https://github.com/marl0ny/1D-Quantum-Mechanics-Applet)
+ [Evidence-Based Medicine](https://github.com/ebmdatalab/)
+ [Chatistics](https://github.com/MasterScrat/Chatistics) - convert chat logs into Panda DataFrames
+ [Pylo](https://github.com/sebdumancic/pylo2) - python front end to Prolog 
+ [Array Programming with Numpy](https://www.nature.com/articles/s41586-020-2649-2)
+ [Opytimizer](https://github.com/gugarosa/opytimizer) - a Nature-Inspired Python Optimizer
+ [Turn images into geometric primitives](https://github.com/Tw1ddle/geometrize) - [demo](https://www.geometrize.co.uk/)
+ [large Zip archives](https://github.com/BuzonIO/zipfly#zipfly)
+ [Microbial simulator](https://github.com/Emergent-Behaviors-in-Biology/community-simulator)
+ [ML for microbiome classification](https://www.biorxiv.org/content/10.1101/816090v1.full), [github repo](https://github.com/SchlossLab/Topcuoglu_ML_XXX_2019/)

+ [DeepMind for Science](https://deepmind.com/blog/article/AlphaFold-Using-AI-for-scientific-discovery), [github repo](https://github.com/deepmind/deepmind-research)

+ [Inline C in Python](https://github.com/georgek42/inlinec)

+ [command-line journal application](https://github.com/jrnl-org/jrnl)

+ [Seq — a language for bioinformatics](https://seq-lang.org/)

+ [API-less video downloader](https://github.com/althonos/InstaLooter)
+ [Python for Feature Film](https://www.gfx.dev/python-for-feature-film)
+ [AeroPython](https://github.com/barbagroup/AeroPython)
+ [Mario](https://github.com/python-mario/mario) - python pipelines for your shell
+ [Pyp](https://github.com/hauntsaninja/pyp) - python from shell

+ [Converting tapes](https://www.joe0.com/2020/10/07/converting-utzoo-wiseman-netnews-archive-to-postgresql-using-python-3-8/) of  [Usenet archives](UsenetArchives.com)

+ [Note](https://github.com/wsw70/note) - command-line note-taking app
+ [predict Personality type based on Reddit profile](https://gimmeserendipity.com/mbtimodel/reddit/)

+ [3D engine ever implemented in DNA code](https://github.com/pallada-92/dna-3d-engine)
+ [Zipfly](https://github.com/BuzonIO/zipfly#lib) - zip archive generator

##### Decentralized Communities
+ [Hummingbard](https://hummingbard.com/hummingbard/introducing-hummingbard) built on top of [Matrix](https://github.com/matrix-org)

### UI

+ PyQT5, [Python GUI](https://build-system.fman.io/pyqt5-tutorial)

+ Apprise, [Push notifications](https://github.com/caronc/apprise/#showhn-one-last-time)

+ [D-Tale](https://github.com/man-group/dtale) - tool to visualize pandas dataframes
+ [dataclass container](https://github.com/joshlk/dataclassframe)


(Slicing, Indexing, Subsetting ataframes)[https://datacarpentry.org/python-ecology-lesson/03-index-slice-subset/]

    df[~((df.A == 0) & (df.B == 2) & (df.C == 6) & (df.D == 0))]  
    df.ix[rows]
    df[((df.A == 0) & (df.B == 2) & (df.C == 6) & (df.D == 0))]
    df.loc[[0,2,4]]
    df.loc[1:3]
    df.iloc[0:df[df.year == 'y3'].index[0]]

### Testing

pip install selenium  # Downloading Python bindings for Selenium
(for windows: C:\Python35\Scripts\pip.exe install selenium)

Place drivers in /usr/bin or /usr/local/bin

|Browser	| Popular Drivers                                                       	|---	|
|---------	|-----------------------------------------------------------------------	|---	|
| Chrome  	| https://sites.google.com/a/chromium.org/chromedriver/downloads        	|   	| 
| Edge    	| https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/ 	|   	|
| Firefox 	| https://github.com/mozilla/geckodriver/releases                       	|   	|  
| Safari  	| https://webkit.org/blog/6900/webdriver-support-in-safari-10/          	|   	| 

Cons: Selenium tests are unstable, time to maintain and run, low ROI

Old school: Ranorex, LeanFT, TestComplete, Telerik and Sahi; Fantom.js, Mocha, Jasmine and Protractor; Screenster

[Testing with Cucumber and Capybara](https://www.gamesparks.com/blog/automated-testing-with-cucumber-and-capybara/)

[Pytest for data scientists](https://statestitle.com/resource/pytest-for-data-scientists/)

### Datasets

[Google dataset search](https://toolbox.google.com/datasetsearch)

[open source](https://deepmind.com/research/open-source/open-source-datasets/)

[Kaggle datasets](https://www.kaggle.com/datasets)

[The Million Song dataset](http://millionsongdataset.com/)

[A RESTish web API for climate change related data](http://api.carbondoomsday.com); [github](https://github.com/giving-a-fuck-about-climate-change/carbondoomsday)

[Disbiome](https://disbiome.ugent.be/) -- [article](https://bmcmicrobiol.biomedcentral.com/articles/10.1186/s12866-018-1197-5)

[Global average temperatures](https://www.nsstc.uah.edu/climate/)[direct link](http://www.nsstc.uah.edu/data/msu/v6.0beta/tlt/uahncdc_lt_6.0beta5.txt)

[cold and warm episodes by season](http://www.cpc.noaa.gov/products/analysis_monitoring/ensostuff/ensoyears.shtml)

[sea level information](http://sealevel.colorado.edu/)

[Astroquery](https://github.com/astropy/astroquery) - collection of tools to access online Astronomical data

### Security
[Gnupg](https://bitbucket.org/vinay.sajip/python-gnupg/) - Python library which takes care of the internal details and allows its users to generate and manage keys, encrypt and decrypt data, and sign and verify messages. See also 

[Pysa](https://engineering.fb.com/security/pysa/): An open source static analysis tool to detect and prevent security issues in Python code

[github mirror](https://github.com/isislovecruft/python-gnupg)

[HTTPX](https://www.python-httpx.org/)

### Administration
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

### What's coming
[Python 3.10](https://pythoninsider.blogspot.com/2020/11/python-3100a2-is-now-available-for.html)
