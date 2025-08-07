import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

################## change background to white
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)  # RESET ALL GLOBAL STYLES

import matplotlib.pyplot as plt
plt.style.use('default')  # USE DEFAULT STYLE

import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.patches import Patch


# === My Data === from metaanalysis.py
my_data = {
    'Year': [2022, 2023, 2024, 2025],
    'My_Validity': [0.664, 0.551, 0.822, 0.834],     # Unweighted mean 
    'My_Std':      [0.309, 0.339, 0.318, 0.228],     # Unweighted std 
    'My_Validity_w': [0.577, 0.628, 0.603, 0.877],   # Weighted mean
    'My_Std_w':      [0.209, 0.398, 0.305, 0.102]    # Weighted std 
}
df_my = pd.DataFrame(my_data)

# === Full-Benchmark HealthBench Data ===
full_benchmark_data = {
    'Year': [2022, 2023, 2024, 2025],
    'FB_Mean_Score': [0.1554, 0.2, 0.3578, 0.4754],
    'FB_Std_Dev': [0, 0, 0.0441, 0.0694]
}
df_full = pd.DataFrame(full_benchmark_data)

# === Consensus-Only HealthBench Data ===
consensus_data = {
    'Year': [2022, 2023, 2024, 2025],
    'Consensus_Mean': [0.7509, 0.76, 0.9011, 0.931],
    'Consensus_Std': [0, 0, 0.0144, 0.0082]
}
df_consensus = pd.DataFrame(consensus_data)

# === Merge All on 'Year' ===
df_all = df_my.merge(df_full, on='Year').merge(df_consensus, on='Year')

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor='white')

# Color palette
colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Line: This Review
ax.plot(df_all['Year'], df_all['My_Validity'], marker='o', markersize=10,
         linewidth=3, color=colors[0], label='This Review')
ax.fill_between(df_all['Year'],
                 df_all['My_Validity'] - df_all['My_Std'],
                 df_all['My_Validity'] + df_all['My_Std'],
                 alpha=0.2, color=colors[0])


##########################  adding weighted values
# New color for weighted curve
weighted_color = '#9b59b6'  # Purple

# Line: This Review (Weighted) - dotted line
ax.plot(df_all['Year'], df_all['My_Validity_w'], marker='D', markersize=9,
        #linewidth=3, color=weighted_color, linestyle='--', label='This Review (Weighted)')
         linewidth=3, color=weighted_color, label='This Review (Weighted)')
ax.fill_between(df_all['Year'],
                df_all['My_Validity_w'] - df_all['My_Std_w'],
                df_all['My_Validity_w'] + df_all['My_Std_w'],
                alpha=0.2, color=weighted_color)

##########################  adding weighted values


# Line: Full-Benchmark
ax.plot(df_all['Year'], df_all['FB_Mean_Score'], marker='s', markersize=10,
         linewidth=3, color=colors[1], label='HealthBench, Full-Benchmark')
ax.fill_between(df_all['Year'],
                 df_all['FB_Mean_Score'] - df_all['FB_Std_Dev'],
                 df_all['FB_Mean_Score'] + df_all['FB_Std_Dev'],
                 alpha=0.2, color=colors[1])

# Line: Consensus-Only
ax.plot(df_all['Year'], df_all['Consensus_Mean'], marker='^', markersize=10,
         linewidth=3, color=colors[2], label='HealthBench, Consensus-Only')
ax.fill_between(df_all['Year'],
                 df_all['Consensus_Mean'] - df_all['Consensus_Std'],
                 df_all['Consensus_Mean'] + df_all['Consensus_Std'],
                 alpha=0.2, color=colors[2])

# Axes and labels
ax.set_xlabel('Year', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Score', fontsize=14, fontweight='bold', labelpad=15)
ax.set_title('Performance Comparison (2022-2025)', fontsize=18, fontweight='bold', pad=20)

# Format y-axis as percentage no longer needed
#\ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

#######################################################
# === Add external weighted points - disclaimer - for 2024 and 2025 ===
external_years = [2022, 2023, 2024, 2025]
external_vals = [57.7448482, 62.16147595,49.14362752, 56.5193225]  # Already in percent


# Convert to proportion scale (0–1.0)
external_vals = [v / 100 for v in external_vals]

# Use same style as "This Review (Weighted)"
ax.plot(external_years, external_vals, marker='D', markersize=9,
        linewidth=3, color=weighted_color, linestyle='--', label='In-study, Weighted (Disclaimer)')



#####################################################

# Ticks
ax.set_xticks(df_all['Year'])
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Limits
ax.set_ylim(0, 1.05)

# Legend
#legend_elements = [
#    Patch(facecolor=colors[0], edgecolor=colors[0], label='This Review', alpha=0.7),
#    Patch(facecolor=colors[1], edgecolor=colors[1], label='HealthBench, Full-Benchmark', alpha=0.7),
#    Patch(facecolor=colors[2], edgecolor=colors[2], label='HealthBench, Consensus-Only', alpha=0.7)
#]

legend_elements = [
    Patch(facecolor=colors[0], edgecolor=colors[0], label='In-Study, Unweighted', alpha=0.7),
    Patch(facecolor=weighted_color, edgecolor=weighted_color, label='In-study, Weighted', alpha=0.7),
    Patch(facecolor=colors[1], edgecolor=colors[1], label='HealthBench, Full-Benchmark', alpha=0.7),
    Patch(facecolor=colors[2], edgecolor=colors[2], label='HealthBench, Consensus-Only', alpha=0.7)
]


#############################
############## adding disclaimer data legend
legend_elements.append(Patch(facecolor=weighted_color, edgecolor=weighted_color,
                             label='In-study, Weighted (Disclaimer)', alpha=0.7))
###########################


# Ensure your data is ready


validity_col = 'Validity, % (Percent accuracy, or truthfulness, or overall quality)'
df['Validity_num'] = pd.to_numeric(df[validity_col], errors='coerce')
df['Sample_Size'] = pd.to_numeric(df['Sample Size (Number of questions/areas to identify/records to process/rankings/respondents/raters)'], errors='coerce')
df.dropna(subset=['Validity_num', 'Sample_Size', 'Year'], inplace=True)

mtick.PercentFormatter(100.0)

#  messing up the scale
#ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))

ax.set_ylim(0, 1.05)  # full 0–100% range
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # format 0.0–1.0 as 0%–100%


ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True,
           facecolor='white', edgecolor='gray', framealpha=0.9)


#### force white background - can turn black after sentiment code
#fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
#fig.patch.set_facecolor('white')       # Set the figure background
#ax.set_facecolor('white')              # Set the plot (axes) background

#plt.style.use('default')



# Annotations for latest values on the right side of the plot
# latest_year = df_all['Year'].iloc[-1]
# for i, col in enumerate(['My_Validity', 'FB_Mean_Score', 'Consensus_Mean']):
#     val = df_all[col].iloc[-1]
#     ax.annotate(f'{val:.1%}', xy=(latest_year, val),
#                 xytext=(10, (-1)**i * 15), textcoords='offset points',
#                 fontsize=12, fontweight='bold', color=colors[i],
#                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=colors[i], alpha=0.7))


# Citation
plt.figtext(0.99, 0.01, 'Gabashvili, 2025', ha='right', va='bottom',
            fontsize=8, color='gray')

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np

# — assume df is your filtered DataFrame with:
#    df['Validity_num']  as numeric percents (e.g. 56.0, 59.0, …)
#    df['Sample_Size']   as integer Ns
#    df['Year']          as int years

# 1) Unweighted mean & std dev by year (in percent)
summary = df.groupby('Year')['Validity_num'].agg(['mean','std']).reset_index()
summary.columns = ['Year','Unweighted_Mean','Std_Dev']

# 2) Weighted mean by year (in percent)
df['Successes'] = df['Validity_num']/100 * df['Sample_Size']
weighted = df.groupby('Year').agg(
    total_successes=('Successes','sum'),
    total_N=('Sample_Size','sum')
).reset_index()
weighted['Weighted_Mean'] = weighted['total_successes']/weighted['total_N'] * 100

# 3) Merge summaries
summary = summary.merge(weighted[['Year','Weighted_Mean']], on='Year')

# 4) Print nicely
for _, row in summary.iterrows():
    print(f"{int(row['Year'])}: "
          f"Unweighted = {row['Unweighted_Mean']:.1f}%, "
          f"StdDev = {row['Std_Dev']:.1f}%, "
          f"Weighted = {row['Weighted_Mean']:.1f}%")
