import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mtick

# Sentiment visualization for papers - two plots
data = {
    'Year': [2022, 2023, 2024, 2025],
    'Cautionary': [3, 7, 4, 1],
    'Contrasting': [0, 3, 0, 1],
    'Negative': [7, 4, 0, 0],
    'Positive': [4, 19, 8, 2],
    'Satisfactory': [11, 8, 0, 0]
}
df = pd.DataFrame(data)

# Set up the figure with dark theme
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 10), dpi=300)
fig.patch.set_facecolor('#121212')

# Calculate totals for percentages and create normalized dataframe
df_totals = df.iloc[:, 1:].sum(axis=1)
df_percent = df.iloc[:, 1:].div(df_totals, axis=0) * 100

# Create custom color palette
colors = {
    'Cautionary': '#FF9500',    # Orange
    'Contrasting': '#9B59B6',   # Purple
    'Negative': '#E74C3C',      # Red
    'Positive': '#2ECC71',      # Green
    'Satisfactory': '#3498DB'   # Blue
}

# Create a grid for subplots
gs = fig.add_gridspec(3, 4, height_ratios=[1, 3, 1])

# 1. MAIN VISUALIZATION: Stacked area chart (top)
ax1 = fig.add_subplot(gs[1, :])

# Create stacked area chart with custom colors
categories = df.columns[1:]
x = df['Year']
y_stack = np.vstack([df_percent[cat] for cat in categories])
labels = list(categories)

# Plot the stacked area
ax1.stackplot(x, y_stack, labels=labels, colors=[colors[cat] for cat in categories], alpha=0.8)

# Add glow effect to the lines
for i, cat in enumerate(categories):
    y_values = np.sum(y_stack[:i+1], axis=0)
    line = ax1.plot(x, y_values, color=colors[cat], linewidth=2.5)
    line[0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.2),
                             path_effects.Normal()])

# Customize the stacked area chart
ax1.set_xlim(2022, 2025)
ax1.set_ylim(0, 100)
ax1.set_xticks(df['Year'])
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_ylabel('Percentage Distribution', fontsize=14, fontweight='bold')
ax1.grid(color='#333333', linestyle='--', alpha=0.3)

# Add title with glow effect
title = ax1.set_title('Sentiment of LLM Evaluation Papers by Model Release Year', fontsize=20, fontweight='bold', pad=20)
title.set_path_effects([path_effects.Stroke(linewidth=3, foreground='#333333'),
                       path_effects.Normal()])

# 2. ABSOLUTE VALUES: Bubble chart (bottom)
ax2 = fig.add_subplot(gs[2, :])

# Create bubble chart for absolute values
bubble_sizes = []
bubble_colors = []
bubble_labels = []
bubble_x = []
bubble_y = []

for year_idx, year in enumerate(df['Year']):
    for cat_idx, category in enumerate(categories):
        value = df.loc[df['Year'] == year, category].values[0]
        if value > 0:  # Only show bubbles for non-zero values
            bubble_x.append(year)
            bubble_y.append(cat_idx)
            bubble_sizes.append(value * 50)  # Scale for visibility
            bubble_colors.append(colors[category])
            bubble_labels.append(str(value))

# Plot bubbles
scatter = ax2.scatter(bubble_x, bubble_y, s=bubble_sizes, c=bubble_colors, alpha=0.7, edgecolors='white', linewidths=1)

# Add value labels to bubbles
for i in range(len(bubble_x)):
    ax2.annotate(bubble_labels[i], (bubble_x[i], bubble_y[i]), 
                 ha='center', va='center', fontweight='bold', color='white')

# Customize bubble chart
ax2.set_yticks(range(len(categories)))
ax2.set_yticklabels(categories, fontsize=12)
ax2.set_xticks(df['Year'])
ax2.set_xlim(2021.5, 2025.5)
ax2.set_ylim(-0.5, len(categories)-0.5)
ax2.set_title('Absolute Counts by Sentiment Category', fontsize=14, fontweight='bold')
ax2.grid(False)

# 3. TREND INDICATORS: Arrow indicators (top)
ax3 = fig.add_subplot(gs[0, :])
ax3.axis('off')

# Calculate trend indicators (comparing 2022 to 2025)
trends = {}
for category in categories:
    start_val = df.loc[df['Year'] == 2022, category].values[0]
    end_val = df.loc[df['Year'] == 2025, category].values[0]
    
    if start_val == 0:
        if end_val == 0:
            trends[category] = "→"  # No change
        else:
            trends[category] = "↑"  # Increase from zero
    else:
        if end_val == 0:
            trends[category] = "↓"  # Decrease to zero
        else:
            percent_change = ((end_val - start_val) / start_val) * 100
            if percent_change > 20:
                trends[category] = "↑"  # Significant increase
            elif percent_change < -20:
                trends[category] = "↓"  # Significant decrease
            else:
                trends[category] = "→"  # No significant change

# Display trend indicators
x_positions = np.linspace(0.1, 0.9, len(categories))
for i, category in enumerate(categories):
    # Add category name
    text = ax3.text(x_positions[i], 0.7, category, 
                   ha='center', va='center', fontsize=14, fontweight='bold', color=colors[category])
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='#333333')])
    
    # Add trend arrow
    arrow = trends[category]
    arrow_color = '#2ECC71' if arrow == "↑" else '#E74C3C' if arrow == "↓" else '#FFFFFF'
    arrow_text = ax3.text(x_positions[i], 0.3, arrow, 
                         ha='center', va='center', fontsize=30, color=arrow_color, fontweight='bold')
    arrow_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='#333333')])

# Add a key insight annotation
#key_insight = "Key Insight: Positive sentiment peaked in 2023, while Negative sentiment disappeared by 2024."
#fig.text(0.5, 0.05, key_insight, ha='center', fontsize=12, fontweight='bold', 
#         bbox=dict(facecolor='#333333', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))

# Add data source and methodology note
fig.text(0.02, 0.02, "Gabashvili, 2025", 
         fontsize=8, color='#999999')

# Add legend with custom styling
legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=5, frameon=True, fontsize=12)
frame = legend.get_frame()
frame.set_facecolor('#333333')
frame.set_edgecolor('white')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.show()

##########################################################################
# Combine 2024 and 2025 into a single "2024-2025" category
combined_row = {
    'Year': '2024-2025',
    'Cautionary': df.loc[df['Year'].isin([2024, 2025]), 'Cautionary'].sum(),
    'Contrasting': df.loc[df['Year'].isin([2024, 2025]), 'Contrasting'].sum(),
    'Negative': df.loc[df['Year'].isin([2024, 2025]), 'Negative'].sum(),
    'Positive': df.loc[df['Year'].isin([2024, 2025]), 'Positive'].sum(),
    'Satisfactory': df.loc[df['Year'].isin([2024, 2025]), 'Satisfactory'].sum()
}

# Build combined dataframe
# df_combined = df[~df['Year'].isin([2024, 2025])].copy()  deprecated
#df_combined = df_combined.append(combined_row, ignore_index=True)   deprecated

# build the combined summary row
combined_row = {
    'Year': '2024-2025',
    'Cautionary': df.loc[df['Year'].isin([2024, 2025]), 'Cautionary'].sum(),
    'Contrasting': df.loc[df['Year'].isin([2024, 2025]), 'Contrasting'].sum(),
    'Negative': df.loc[df['Year'].isin([2024, 2025]), 'Negative'].sum(),
    'Positive': df.loc[df['Year'].isin([2024, 2025]), 'Positive'].sum(),
    'Satisfactory': df.loc[df['Year'].isin([2024, 2025]), 'Satisfactory'].sum()
}

# filter out 2024 & 2025, then concat the new row
base = df.loc[~df['Year'].isin([2024, 2025])].copy()
df_combined = pd.concat(
    [base, pd.DataFrame([combined_row])],
    ignore_index=True
)

#print(df_combined)



# Prepare normalized percentages
categories = ['Cautionary', 'Contrasting', 'Negative', 'Positive', 'Satisfactory']
totals = df_combined[categories].sum(axis=1)
df_percent = df_combined[categories].div(totals, axis=0) * 100

# Define colors
colors = {
    'Cautionary': '#FF9500',
    'Contrasting': '#9B59B6',
    'Negative': '#E74C3C',
    'Positive': '#2ECC71',
    'Satisfactory': '#3498DB'
}

# Plot combined stacked area chart
plt.style.use('dark_background')
plt.figure(figsize=(10, 6), dpi=300)
x = np.arange(len(df_combined))
y_stack = np.vstack([df_percent[cat] for cat in categories])

plt.stackplot(x, y_stack, labels=categories, colors=[colors[cat] for cat in categories], alpha=0.8)

# Glow effect
for i, cat in enumerate(categories):
    y_vals = np.sum(y_stack[:i+1], axis=0)
    line = plt.plot(x, y_vals, color=colors[cat], linewidth=2.5)
    line[0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.2),
                              path_effects.Normal()])

plt.xticks(x, df_combined['Year'])
plt.yticks(fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel('Model Release Year', fontsize=14, fontweight='bold')
plt.ylabel('Percentage Distribution', fontsize=14, fontweight='bold')
plt.title('Sentiment of LLM Evaluation Papers (2022, 2023, 2024–2025)', fontsize=16, fontweight='bold', pad=15)
plt.grid(color='#333333', linestyle='--', alpha=0.3)
plt.legend(loc='upper center', ncol=5, frameon=True, fontsize=10)
plt.tight_layout()
plt.show()
