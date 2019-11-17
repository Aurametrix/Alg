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
