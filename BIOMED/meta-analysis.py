import numpy as np

# Clean and convert the relevant columns
data['sample_size'] = pd.to_numeric(data['sample_size'], errors='coerce')
data['score'] = data['score'].str.replace('%', '').astype(float) / 100  # Convert score to a decimal

# Drop rows with missing or invalid data in these columns
data = data.dropna(subset=['sample_size', 'score'])

# Compute effect size and precision
data['effect_size'] = data['score'] * data['sample_size']
data['standard_error'] = 1 / np.sqrt(data['sample_size'])  # Assuming this relation
data['precision'] = 1 / data['standard_error']

# Display updated dataset
#data[['aspect', 'sample_size', 'score', 'effect_size', 'precision']].head()

# Aggregate effect sizes by broader categories
aggregated_data = data.groupby('broader_category').agg(
    total_effect_size=('effect_size', 'sum'),
    mean_effect_size=('effect_size', 'mean'),
    std_effect_size=('effect_size', 'std'),
    count=('effect_size', 'size')
).reset_index()

# Visualize aggregated effect sizes by broader categories
plt.figure(figsize=(12, 8))
plt.barh(aggregated_data['broader_category'], aggregated_data['total_effect_size'], edgecolor='black', alpha=0.7)
plt.title('Total Effect Size by Broader Categories', fontsize=16)
plt.xlabel('Total Effect Size', fontsize=14)
plt.ylabel('Broader Categories', fontsize=14)
plt.grid(axis='x', alpha=0.75)
plt.show()

# Display the aggregated data for reference
import ace_tools as tools; tools.display_dataframe_to_user(name="Aggregated Effect Sizes by Broader Categories", dataframe=aggregated_data)

# Cochran's Q and I² statistics calculations

# Compute the weighted mean effect size
weighted_mean_effect_size = (
    data['effect_size'] / data['standard_error']**2
).sum() / (1 / data['standard_error']**2).sum()

# Compute Cochran's Q
data['squared_diff'] = (
    ((data['effect_size'] - weighted_mean_effect_size)**2) / data['standard_error']**2
)
cochrans_q = data['squared_diff'].sum()

# Compute degrees of freedom (number of studies - 1)
degrees_of_freedom = len(data) - 1

# Compute I² statistic
i_squared = max(0, (cochrans_q - degrees_of_freedom) / cochrans_q) * 100

# Output results
{
    "Cochran's Q": cochrans_q,
    "Degrees of Freedom": degrees_of_freedom,
    "I² (%)": i_squared
}

# Analyze precision trends over sample sizes
import seaborn as sns
plt.figure(figsize=(10, 6))

# Create a scatter plot to analyze the trend
sns.scatterplot(x=filtered_data['sample_size'], y=filtered_data['precision'])

# Add a line of best fit to visualize trends
sns.regplot(x=filtered_data['sample_size'], y=filtered_data['precision'], scatter=False, color='red', ci=None)

# Add titles and labels
plt.title('Precision Trends Over Sample Sizes', fontsize=16)
plt.xlabel('Sample Size', fontsize=14)
plt.ylabel('Precision', fontsize=14)

plt.grid(True)
plt.show()

# -----------------------------------------------
# extract all LLM/transformers analyzed in each paper
import pandas as pd

# Read the CSV file into a pandas DataFrame
#df = pd.read_csv('data.csv')
#df = pd.read_csv('data.csv', encoding='latin-1')
df = pd.read_csv('data.csv', encoding='iso-8859-1')
#df = pd.read_csv('data.csv', engine='python')

# Group by 'Reference' and aggregate LLM columns as a list (without duplicates)
def combine_llms(group):
    llms = []
    for col in ['LLM1', 'LLM2', 'LLM3']:
        for llm in group[col].dropna().tolist():
            if llm not in llms:  # Add only if not already in the list
                llms.append(llm)
    return llms

refs_df = df.groupby('Reference').apply(combine_llms).reset_index(name='LLMs')

# Export the results to a new CSV file
refs_df.to_csv('refs.csv', index=False)
print("File 'refs.csv' has been created successfully.")
