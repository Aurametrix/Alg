#!/usr/bin/env python
from numpy import *
import Gnuplot

g = Gnuplot.Gnuplot()
g.title('My Systems Plot')
g.xlabel('Date')
g.ylabel('Response')
g('set auto x')
g('set term png')
g('set out "output.png"')
g('set timefmt "%s"')
g('set xdata time')
g('set xtic rotate by 45 scale 1 font ",2"')
g('set key noenhanced')
g('set format x "%H:%M:%S"')
g('set grid')


databuff = Gnuplot.File("repo", using='1:2',title="test")
g.plot(databuff)

