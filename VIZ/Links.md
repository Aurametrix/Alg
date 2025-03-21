[PyVIZ](https://pyviz.org/overviews/index.html) - The Python visualization landscape

[Visualize terminal sessions as SVG animations](https://github.com/nbedos/termtosvg)

[The Python Graph Gallery](https://python-graph-gallery.com/boxplot/) - from the [Catalogue ](https://datavizcatalogue.com/index.html)

[Interactive Plotting Library](https://github.com/JetBrains/lets-plot/blob/master/README_PYTHON.md)

[Charts, Graphs, Plots with Altair](https://tech.marksblogg.com/python-data-visualisation-charts-graphs-plots.html)

[Data Structure Visualizations](https://www.cs.usfca.edu/~galles/visualization/Algorithms.html)

[Pybaobab](https://gitlab.tue.nl/20040367/pybaobab) - vizualizing decision trees

[Debug visualizer for VS code](https://github.com/hediet/vscode-debug-visualizer)

[Algorithm visualizer](https://algorithms.laszlokorte.de/)

[Circular viz](https://github.com/moshi4/pyCirclize)

[Kroki](https://github.com/yuzutech/kroki); provides Diagrams plugin for ChatGPT4

[Tools for visualizing a codebase](https://lmy.medium.com/7-tools-for-visualizing-a-codebase-41b7cddb1a14)
[Tree viz](https://treevis.net/); [paper](https://ieeexplore.ieee.org/document/6056510)

[Python & OpenGL for Scientific Visualization](https://www.labri.fr/perso/nrougier/python-opengl/)

[Pygwalker](https://github.com/Kanaries/pygwalker) - Python binding of Graphic Walker, turns Your Pandas Dataframe into a Tableau-Style UI for Visual Analysis; [in Kaggle](https://www.kaggle.com/asmdef/pygwalker-test)

[DeTikZify](https://github.com/potamides/DeTikZify) - multimodal language model that automatically synthesizes scientific figures as semantics-preserving TikZ graphics programs based on sketches and existing figures

### VosViewer
+ [tutorial](https://seinecle.github.io/vosviewer-tutorials/generated-html/importing-en.html), [pdf](https://seinecle.github.io/vosviewer-tutorials/generated-pdf/importing-en.pdf)
+ [from freely available citation data](https://www.cwts.nl/blog?article=n-r2r294)
+ [format converter](https://nocodefunctions.com/networkconverter/network_format_converter.html)

### text-to-image
+ [imagen](https://gweb-research-imagen.appspot.com/)
+ [open source DALLE-2 implementation](https://github.com/lucidrains/DALLE2-pytorch)


Metaknowledge, a bibliometric toolkit in Python

import metaknowledge as mk
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
import metaknowledge.contour as mkv
import pandas
RC = mk.RecordCollection("pubmed_medline.txt")
for R in RC:
  if 'AB' in R.keys():
  print(R['AB'])
  print('\n')


