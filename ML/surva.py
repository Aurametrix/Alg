# Import required libraries :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

#Read the dataset :
data = pd.read_csv("lung.csv")
print (data.head())

# Organize our data :
# If status = 1 , then dead = 0
# If status = 2 , then dead = 1
data.loc[data.status == 1, 'dead'] = 0
data.loc[data.status == 2, 'dead'] = 1
print (data.head())


# kmf_m for male data.
# kmf_f for female data.
kmf_m = KaplanMeierFitter() 
kmf_f = KaplanMeierFitter() 

# Dividing data into groups :
Male = data.query("sex == 1")
Female = data.query("sex == 2")

# The 1st arg accepts an array or pd.Series of individual survival times
# The 2nd arg accepts an array or pd.Series that indicates if the event 
# interest (or death) occured.
kmf_m.fit(durations =  Male["time"],event_observed = Male["dead"] ,label="Male")
kmf_f.fit(durations =  Female["time"],event_observed = Female["dead"], label="Female")
print (kmf_m.event_table)
print (kmf_f.event_table)

print (kmf_m.predict(11))
print (kmf_f.predict(11))

print (kmf_m.survival_function_)
print (kmf_f.survival_function_)

# Plot the survival_function data :
kmf_m.plot()
kmf_f.plot()
plt.xlabel("Days passed")
plt.ylabel("Survival")
plt.title("KMF")


print (kmf_m.cumulative_density_)
print (kmf_f.cumulative_density_)

kmf_m.plot_cumulative_density()
kmf_f.plot_cumulative_density()

# Hazard FUnction :
from lifelines import NelsonAalenFitter
naf_m = NelsonAalenFitter()
naf_f = NelsonAalenFitter()

naf_m.fit(Male["time"],event_observed = Male["dead"])
naf_f.fit(Female["time"],event_observed = Female["dead"])

print (naf_m.cumulative_hazard_)
print (naf_f.cumulative_hazard_)

naf_m.plot_cumulative_hazard()
naf_f.plot_cumulative_hazard()

# We can predict the value of a certain point :
naf_m.predict(1022)
naf_f.predict(1022)

# Log-Rank Test

# Define variables :
T=Male['time']
E=Male['dead']
T1=Female['time']
E1=Female['dead']

from lifelines.statistics import logrank_test

results=logrank_test(T,T1,event_observed_A=E, event_observed_B=E1)
results.print_summary()
