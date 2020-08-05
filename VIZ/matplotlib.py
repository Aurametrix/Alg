from pylab import *
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
T = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(T,T)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

df = pd.read_excel("https://github.com/chris1610/pbpython/blob/master/data/sample-salesv3.xlsx?raw=true")
df.head()

top_10 = (df.groupby('name')['ext price', 'quantity'].agg({'ext price': 'sum', 'quantity': 'count'})
          .sort_values(by='ext price', ascending=False))[:10].reset_index()
top_10.rename(columns={'name': 'Name', 'ext price': 'Sales', 'quantity': 'Purchases'}, inplace=True)

plt.style.available

['seaborn-dark',
 'seaborn-dark-palette',
 'fivethirtyeight',
 'seaborn-whitegrid',
 'seaborn-darkgrid',
 'seaborn',
 'bmh',
 'classic',
 'seaborn-colorblind',
 'seaborn-muted',
 'seaborn-white',
 'seaborn-talk',
 'grayscale',
 'dark_background',
 'seaborn-deep',
 'seaborn-bright',
 'ggplot',
 'seaborn-paper',
 'seaborn-notebook',
 'seaborn-poster',
 'seaborn-ticks',
 'seaborn-pastel']
 
 plt.style.use('ggplot')
 
 top_10.plot(kind='barh', y="Sales", x="Name")
 
 
fig, ax = plt.subplots()
top_10.plot(kind='barh', y="Sales", x="Name", ax=ax)

fig, ax = plt.subplots()
top_10.plot(kind='barh', y="Sales", x="Name", ax=ax)
ax.set_xlim([-10000, 140000])
ax.set_xlabel('Total Revenue')
ax.set_ylabel('Customer');

fig, ax = plt.subplots()
top_10.plot(kind='barh', y="Sales", x="Name", ax=ax)
ax.set_xlim([-10000, 140000])
ax.set(title='2014 Revenue', xlabel='Total Revenue', ylabel='Customer')

import numpy as np
import matplotlib.pyplot as plt

np.x1=np.array([5,6,7,8])  # Xs of BEFORE  Bad bacteria
np.y1=np.array([1,2,3,4]) # Ys of BEFORE Good bacteria
np.x2=np.array([0,1,3,2]) #Xs of AFTER - BAD
np.y2=np.array([0,0,2,2]) #Ys of AFTER - GOOD

plt.figure()
ax = plt.gca()
ax.quiver(np.x1, np.y1, np.x2-np.x1, np.y2-np.y1, angles='xy', scale_units='xy', scale=1)

                      
                         
ax.set_xlim([-1, 20])
ax.set_ylim([-1, 15])
plt.draw()
plt.show()
