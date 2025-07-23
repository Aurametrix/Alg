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

# T-test for Diagnosis vs Education
diag = df_cats[df_cats['Category_Simple'] == 'Diagnosis']['Validity']
edu = df_cats[df_cats['Category_Simple'] == 'Education']['Validity']
t_stat, p_val = stats.ttest_ind(diag, edu, equal_var=False)

print("\nT-test - Diagnosis vs Education:")
print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.3g}")
