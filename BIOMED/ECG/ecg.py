#import all the rquired packages
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#load/read the ECG signal which is in excel format
a = pd.read_excel(r"C:\Users\LAVANYA\Documents\Python Bootcamp\project\ECG_data.xlsx")

#converting the signal to 1D array
b = np.array(a)
c = b.transpose()

#to remove the extra square braces from the signal valuesecg.py
d = reduce(lambda x,y: x+y,c)

#segmenting the signal to one minute each to find the accurate Heart rate
d1 = d[0:12000]
d2 = d[12000:24000]
d3 = d[24000:36000]
d4 = d[36000:48000]
d5 = d[48000:60000]

#find the peaks of the signal,Heart Rate and plot it
m1,n1 = find_peaks(d1, distance=125)
HR1 = int(len(m1))
m2,n2 = find_peaks(d2, distance=125)
HR2 = int(len(m2))
m3,n3 = find_peaks(d3, distance=125)
HR3 = int(len(m3))
m4,n4 = find_peaks(d4, distance=125)
HR4 = int(len(m4))
m5,n5 = find_peaks(d5, distance=125)
HR5 = int(len(m5))
HR = (HR1+HR2+HR3+HR4+HR5)/5
print("Heart Rate(using ECG)=",int(HR),"BPM")

#Plot the signal with located peaks
e = d[72000:84000]
m6,n6 = find_peaks(e, distance=125)
plt.figure()
plt.subplot(2,2,1)
plt.plot(m6, e[m6],"^")
plt.plot(e)
plt.title("One minute plot of ECG signal")
m,n = find_peaks(d, distance=125)
plt.subplot(2,2,2)
plt.plot(d)
plt.plot(m, d[m],"^")
plt.title("10 minutes ECG signal")



#same procedure is applied to PPG
p = pd.read_excel(r"C:\Users\LAVANYA\Documents\Python Bootcamp\project\PPG_data.xlsx")

#reading the sinal as array and transposing it
q = np.array(p)
r = q.transpose()

#to remove the extra square braces from the signal values
s = reduce(lambda x,y: x+y,r)
s1 = s[0:12000]
s2 = s[12000:24000]
s3 = s[24000:36000]
s4 = s[36000:48000]
s5 = s[48000:60000]

#find the peaks of the signal,Heart Rate and plot it
u1,v1 = find_peaks(s1, distance=100)
Hr1 = int(len(u1))
u2,v2 = find_peaks(s2, distance=100)
Hr2 = int(len(u2))
u3,v3 = find_peaks(s3, distance=100)
Hr3 = int(len(u3))
u4,v4 = find_peaks(s4, distance=100)
Hr4 = int(len(u4))
u5,v5 = find_peaks(s5, distance=100)
Hr5 = int(len(u5))
Hr = (Hr1+Hr2+Hr3+Hr4+Hr5)/5
print("Heart Rate(using PPG)=",int(Hr),"BPM")

#find the peaks of the signal,Heart Rate and plot it
t = s[84000:96000]
u6,v6 = find_peaks(t, distance=100)
plt.subplot(2,2,3)
plt.plot(t)
plt.plot(u6, t[u6],"^")
plt.title("One minute plot of PPG signal")
u,v = find_peaks(s, distance=100)
plt.subplot(2,2,4)
plt.plot(s)
plt.plot(u, s[u],"^")
plt.title("10 minutes PPG signal")
plt.show()
