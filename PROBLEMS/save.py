a_file = open("test.txt", "w")
for row in an_array:
    np.savetxt(a_file, row)

a_file.close()

import numpy as np
import pandas as pd

np.random.seed(42)

a = np.random.randn(3, 4)
a[2][2] = np.nan
print a
np.savetxt('np.csv', a, fmt='%.2f', delimiter=',', header=" #1,  #2,  #3,  #4")
df = pd.DataFrame(a)
print df
df.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
