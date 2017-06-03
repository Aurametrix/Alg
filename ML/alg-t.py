import quandl 
aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")

# Import `numpy` as `np`
import numpy as np

# Assign `Adj Close` to `daily_close`
daily_close = aapl[['___________']]

# Daily returns
daily_pct_change = daily_close.__________()

# Replace NA values with 0
daily_pct_change.fillna(0, inplace=True)

# Inspect daily returns
print(______________)

# Daily log returns
daily_log_returns = np.log(daily_close.pct_change()+1)

# Print daily log returns
print(daily_log_returns)
