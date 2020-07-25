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

def calculate_pvalue(data, exposure, outcome):
    rr = RiskRatio()
    rr.fit(data, exposure=exposure, outcome=outcome)

    # calculating p-value
    est = rr.results['RiskRatio'][1]
    std = rr.results['SD(RR)'][1]
    z_score = np.log(est) / std
    p_value = norm.sf(abs(z_score)) * 2
    return est, p_value
