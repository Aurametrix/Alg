# see also https://github.com/LSYS/forestplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example data for systematic review meta-analysis
data = {
    'Study': ['Study A', 'Study B', 'Study C', 'Study D', 'Study E'],
    'Effect_Size': [0.3, -0.1, 0.25, 0.4, 0.1],
    'Lower_CI': [0.1, -0.25, 0.05, 0.2, -0.1],
    'Upper_CI': [0.5, 0.05, 0.45, 0.6, 0.3]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting forest plot
plt.figure(figsize=(8, 6))

# Create a horizontal line for each study (effect size and confidence interval)
for index, row in df.iterrows():
    plt.plot([row['Lower_CI'], row['Upper_CI']], [index, index], color='black')
    plt.plot(row['Effect_Size'], index, 'o', color='red')

# Add study labels
y_ticks = np.arange(len(df))
plt.yticks(y_ticks, df['Study'])
plt.axvline(x=0, color='gray', linestyle='--')  # Reference line at zero

# Add labels
plt.xlabel('Effect Size')
plt.title('Forest Plot for Meta-Analysis')

# Show plot
plt.tight_layout()
plt.show()
