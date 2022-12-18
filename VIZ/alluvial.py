import alluvial
import matplotlib.pyplot as plt
import numpy as np

input_data = {'a': {'aa': 0.3, 'cc': 0.7,},
              'b': {'aa': 2, 'bb': 0.5,},
              'c': {'aa': 0.5, 'bb': 0.5, 'cc': 1.5,}}

ax = alluvial.plot(input_data)
fig = ax.get_figure()
fig.set_size_inches(5,5)
plt.show()
