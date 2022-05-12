[Visualize terminal sessions as SVG animations](https://github.com/nbedos/termtosvg)

[The Python Graph Gallery](https://python-graph-gallery.com/boxplot/) - from the [Catalogue ](https://datavizcatalogue.com/index.html)

[Interactive Plotting Library](https://github.com/JetBrains/lets-plot/blob/master/README_PYTHON.md)

[Data Structure Visualizations](https://www.cs.usfca.edu/~galles/visualization/Algorithms.html)

[Pybaobab](https://gitlab.tue.nl/20040367/pybaobab) - vizualizing decision trees

[Debug visualizer for VS code](https://github.com/hediet/vscode-debug-visualizer)

[Algorithm visualizer](https://algorithms.laszlokorte.de/)

[Tools for visualizing a codebase](https://lmy.medium.com/7-tools-for-visualizing-a-codebase-41b7cddb1a14)

VosViewer
https://seinecle.github.io/vosviewer-tutorials/generated-pdf/importing-en.pdf
https://nocodefunctions.com/networkconverter/network_format_converter.html


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


