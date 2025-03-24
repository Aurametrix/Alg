import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


def plot_microbial_species(df, species_name, label_column="LABEL"):  # Removed y_min/y_max
    """
    Plots a bar chart with error bars for the specified microbial species across groups A, R, and C,
    and prints means ± standard deviations.

    Parameters:
        df (pd.DataFrame): The dataset containing microbial species data.
        species_name (str): The name of the species to filter and plot.
        label_column (str): The column name representing group labels (default is 'LABEL').
    """
    df.columns = df.columns.str.strip()
    species_columns = [col for col in df.columns if species_name in col]

    if not species_columns:
        print(f"No columns found for species: {species_name}")
        return

    def group_stats(group_label):
        group_data = df[df[label_column] == group_label][species_columns]
        means = group_data.mean()
        stds = group_data.std()
        errors = stds / np.sqrt(group_data.count())
        return means, stds, errors

    group_A_means, group_A_stds, group_A_errors = group_stats('A')
    group_R_means, group_R_stds, group_R_errors = group_stats('R')
    group_C_means, group_C_stds, group_C_errors = group_stats('C')

    print("\nMean ± Std Dev for each group:")
    for col in species_columns:
        print(f"{col}:")
        print(f"  A: {group_A_means[col]:.2f} ± {group_A_stds[col]:.2f}")
        print(f"  R: {group_R_means[col]:.2f} ± {group_R_stds[col]:.2f}")
        print(f"  C: {group_C_means[col]:.2f} ± {group_C_stds[col]:.2f}")

    x = np.arange(len(species_columns))
    width = 0.25

    all_means = pd.concat([group_A_means, group_R_means, group_C_means])
    y_min = all_means.min() * 0.95
    y_max = all_means.max() * 1.01

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, group_A_means, width, yerr=group_A_errors, capsize=5, label='A', alpha=0.7)
    ax.bar(x, group_R_means, width, yerr=group_R_errors, capsize=5, label='R', alpha=0.7)
    ax.bar(x + width, group_C_means, width, yerr=group_C_errors, capsize=5, label='C', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(species_columns, rotation=90)
    ax.set_ylabel("Average Value")
    ax.set_title(f"Averages of {species_name} for A vs R and C with Error Bars")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


file_path = r"C:\Users\all\Documents\IG\MEBO\Microbiome\uBiome\DATAPROCESSING\PCA\Python Scripts\MEBO_data-withlabels.csv"

# Load the dataset with the detected encoding
df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()

label_column = "LABEL"
numeric_columns = df.select_dtypes(include=['number']).columns

group_A = df[df[label_column] == 'A'][numeric_columns]
group_R = df[df[label_column] == 'R'][numeric_columns]

p_values = {}
for col in numeric_columns:
    if group_A[col].size > 3 and group_R[col].size > 3:
        try:
            stat, p_value = ttest_ind(group_A[col].dropna(), group_R[col].dropna(), equal_var=False)
            if p_value > 0.05:
                stat, p_value = mannwhitneyu(group_A[col].dropna(), group_R[col].dropna(), alternative='two-sided')
        except:
            p_value = None
    else:
        p_value = None
    p_values[col] = p_value

p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'P-Value'])
output_file_path = "p_values_results.csv"
p_values_df.to_csv(output_file_path, index=False)

# Export original columns with p-value < 0.01
significant_features = p_values_df[p_values_df['P-Value'] < 0.01]['Feature'].tolist()
df_significant = df[[label_column] + significant_features]
df_significant.to_csv("significant_features_data.csv", index=False)

# Plot
plot_microbial_species(df, "Bacteria superkingdom")
