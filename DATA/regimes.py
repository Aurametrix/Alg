import numpy as np
import matplotlib.pyplot as plt

# Convert date column to datetime and clean Google Scholar column
pubs_df['Date'] = pd.to_datetime(pubs_df['Date'], format='%d-%b-%y')
pubs_df['Google Scholar'] = pubs_df['Google Scholar'].str.replace(',', '').astype(int)

# Sort by date and compute monthly increments
pubs_df = pubs_df.sort_values('Date').reset_index(drop=True)
pubs_df['Monthly Delta'] = pubs_df['Google Scholar'].diff()

# Calculate a rolling mean of deltas to smooth fluctuations
pubs_df['Rolling Delta'] = pubs_df['Monthly Delta'].rolling(window=2, min_periods=1).mean()

# Plot the original values and the regime-detection line (rolling delta)
plt.figure(figsize=(10, 6))
plt.plot(pubs_df['Date'], pubs_df['Google Scholar'], label='Cumulative Citations', marker='o')
plt.plot(pubs_df['Date'], pubs_df['Rolling Delta'], label='Smoothed Monthly Growth', marker='x')
plt.xlabel("Date")
plt.ylabel("Count")
plt.title("Regime Detection Based on Google Scholar Growth")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Show the DataFrame with computed values for further analysis
pubs_df[['Date', 'Google Scholar', 'Monthly Delta', 'Rolling Delta']]
