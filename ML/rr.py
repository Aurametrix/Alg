import numpy as np
import pandas as pd
from scipy.stats import norm
from zepid import RiskRatio

# creating an example data set
df = pd.DataFrame()
df['A'] = [1, 0, 1, 0, 1, 1]
df['B'] = [1, 1, 0, 0, 0, 0]

# calculating risk ratio
rr = RiskRatio()
rr.fit(df, exposure='A', outcome='B')

# calculating p-value
est= rr.results['RiskRatio'][1]
std = rr.results['SD(RR)'][1]
z_score = np.log(est)/std
p_value = norm.sf(abs(z_score))*2

# Running rr.summary() will print all the different measures calculated to the console. 
# Confidence Interval, CL- Confidence Level
# Since Lnof RR, ~ normally distributed, a confidence interval is generated for Ln(RR), 
# and then the antilog of the upper and lower limits of the confidence interval for Ln(RR) 
# are computed to give the upper and lower limits of the confidence interval for the RR.
# rr.results: the lower CL is rr.results['RR_LCL'] and the upper is rr.results['RR_UCL']

def calculate_pvalue(data, exposure, outcome):
    rr = RiskRatio()
    rr.fit(data, exposure=exposure, outcome=outcome)

    # calculating p-value
    est = rr.results['RiskRatio'][1]
    std = rr.results['SD(RR)'][1]
    z_score = np.log(est) / std
    p_value = norm.sf(abs(z_score)) * 2
    return est, p_value
