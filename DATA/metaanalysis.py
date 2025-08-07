import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load and clean the data
df = pd.read_csv('scores.csv', encoding='ISO-8859-1')

# Rename columns for easier handling
df.rename(columns={
    'Validity, % (Percent accuracy, or truthfulness, or overall quality)': 'Validity',
    'Sample Size (Number of questions/areas to identify/records to process/rankings/respondents/raters)': 'Sample_Size',
    'LLM-category/underlying language model technology': 'LLM_Category',
    'Model_Age (newer, larger) 3 is GPTV,': 'Model_Age'
}, inplace=True)

# Convert relevant columns
df['Validity'] = pd.to_numeric(df['Validity'], errors='coerce')
df = df.dropna(subset=['Validity', 'Year', 'Category'])

# --- Plot: Validity by Year ---
years_of_interest = [2023, 2024, 2025]
df_years = df[df['Year'].isin(years_of_interest)]

plt.figure(figsize=(8, 6))
sns.pointplot(data=df_years, x='Year', y='Validity', ci=95, capsize=.1, join=False, palette='muted')
plt.title('Mean Validity by Year with 95% CI')
plt.ylabel('Mean Validity Score')
plt.xlabel('Year')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('validity_by_year.png')
plt.close()

# Tukey HSD for year comparisons
tukey_years = pairwise_tukeyhsd(endog=df_years['Validity'], groups=df_years['Year'], alpha=0.05)
print("Tukey HSD - Year Comparison:")
print(tukey_years.summary())

# --- Plot: Diagnosis vs Education ---
relevant_cats = [
    'Medical Records and Diagnostic Processes',
    'Patient Education Materials and Readability Studies'
]
df_cats = df[df['Category'].isin(relevant_cats)].copy()
df_cats['Category_Simple'] = df_cats['Category'].map({
    'Medical Records and Diagnostic Processes': 'Diagnosis',
    'Patient Education Materials and Readability Studies': 'Education'
})

plt.figure(figsize=(8, 6))
sns.pointplot(data=df_cats, x='Category_Simple', y='Validity', ci=95, capsize=.1, join=False, palette='Set2')
plt.title('Mean Validity: Diagnosis vs Education with 95% CI')
plt.ylabel('Mean Validity Score')
plt.xlabel('Category')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('validity_by_category.png')
plt.close()

##########################################
### computing average scores for LLMs, excluding reference 81 (disclaimers)
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from itertools import combinations

# Load the CSV with appropriate encoding
file_path = "scores.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Define relevant columns
validity_col = 'Validity, % (Percent accuracy, or truthfulness, or overall quality)'
sample_size_col = 'Sample Size (Number of questions/areas to identify/records to process/rankings/respondents/raters)'

# Step 1: Preprocess
df = df[~df['RF ID'].eq(81)]  # Exclude RF ID 81
df = df[~df['LLM-category'].astype(str).str.startswith('NOT')]  # Exclude LLM-category starting with 'NOT'
df['Year'] = pd.to_datetime(df['Date of version'], errors='coerce').dt.year  # Extract model release year

# Convert to numeric and scale validity
df[validity_col] = pd.to_numeric(df[validity_col], errors='coerce') * 100
df[sample_size_col] = pd.to_numeric(df[sample_size_col], errors='coerce')
df.dropna(subset=[validity_col, sample_size_col, 'Year'], inplace=True)

# Step 2: Compute stats by year
def weighted_stats(group):
    values = group[validity_col]
    weights = group[sample_size_col]
    mean = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
    return pd.Series({
        'unweighted_mean': values.mean(),
        'unweighted_std': values.std(),
        'weighted_mean': mean,
        'weighted_std': std,
        'total_n': weights.sum()
    })

stats_by_year = df.groupby('Year').apply(weighted_stats)



# Step 3: Pairwise significance tests (Welch's t-test)
years = stats_by_year.index.tolist()
pairwise_results = []

for y1, y2 in combinations(years, 2):
    group1 = df[df['Year'] == y1][validity_col]
    group2 = df[df['Year'] == y2][validity_col]
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    pairwise_results.append({
        'Comparison': f"{y1} vs {y2}",
        't-stat': t_stat,
        'p-value': p_val,
        'Significant (p<0.05)': p_val < 0.05
    })

pairwise_df = pd.DataFrame(pairwise_results)

#import ace_tools as tools
#tools.display_dataframe_to_user(name="Validity Statistics by Model Release Year", dataframe=stats_by_year)
#tools.display_dataframe_to_user(name="Pairwise Significance Tests Between Years", dataframe=pairwise_df)

print("=== Validity Statistics by Model Release Year ===")
print(stats_by_year)

print("\n=== Pairwise Significance Tests Between Years by Model Release ===")
print(pairwise_df)




#################################  simpler - by year of publication  - since it was overwritten start anew

# Load the CSV with appropriate encoding
file_path = "scores.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Preserve the original publication year
df['Publication Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Define relevant columns
validity_col = 'Validity, % (Percent accuracy, or truthfulness, or overall quality)'
sample_size_col = 'Sample Size (Number of questions/areas to identify/records to process/rankings/respondents/raters)'

# Step 1: Preprocess - exclude paper$81 and NON LLMs
#df = df[~df['RF ID'].eq(81)]  # Exclude RF ID 81
df = df[~df['LLM-category'].astype(str).str.startswith('NOT')]  # Exclude LLM-category starting with 'NOT'

# Extract model release year into a new column
df['Release Year'] = pd.to_datetime(df['Date of version'], errors='coerce').dt.year

# Convert to numeric and scale validity
df[validity_col] = pd.to_numeric(df[validity_col], errors='coerce') * 100
df[sample_size_col] = pd.to_numeric(df[sample_size_col], errors='coerce')
df.dropna(subset=[validity_col, sample_size_col, 'Release Year', 'Publication Year'], inplace=True)

# === Compute weighted and unweighted averages by publication year (from 'Year' column) ===

df['Publication Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.dropna(subset=['Publication Year'], inplace=True)
# === Compute weighted and unweighted averages by publication year ===
def weighted_publication_stats(group):
    values = group[validity_col]
    weights = group[sample_size_col]
    weighted_mean = np.average(values, weights=weights)
    weighted_std = np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights))
    return pd.Series({
        'Unweighted Mean (%)': values.mean(),
        'Unweighted Std': values.std(),
        'Weighted Mean (%)': weighted_mean,
        'Weighted Std': weighted_std,
        'Total Sample Size': weights.sum()
    })

publication_year_stats = df.groupby('Year').apply(weighted_publication_stats).reset_index()

# Optional: Display or export
# tools.display_dataframe_to_user(name="Validity Statistics by Publication Year", dataframe=publication_year_stats)
print("=== Validity Statistics by Publication Year ===")
print(publication_year_stats)


######################################


# T-test for Diagnosis vs Education
diag = df_cats[df_cats['Category_Simple'] == 'Diagnosis']['Validity']
edu = df_cats[df_cats['Category_Simple'] == 'Education']['Validity']
t_stat, p_val = stats.ttest_ind(diag, edu, equal_var=False)

print("\nT-test - Diagnosis vs Education:")
print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.3g}")
