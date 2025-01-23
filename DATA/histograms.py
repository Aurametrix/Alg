# Quantitative analysis of holding periods (durations) by year
years_analysis = []

for year in years + [2023, 2024]:  # Include 2023 and 2024 for comparison
    year_data = filtered_data[filtered_data['sale_date'].dt.year == year]
    if not year_data.empty:
        stats = {
            'Year': year,
            'Median Duration': year_data['duration_years_rounded'].median(),
            'Mean Duration': year_data['duration_years_rounded'].mean(),
            'Mode Duration': year_data['duration_years_rounded'].mode()[0] if not year_data['duration_years_rounded'].mode().empty else None,
            'Standard Deviation': year_data['duration_years_rounded'].std(),
            'Short-Term Sales (<3 Years)': (year_data['duration_years_rounded'] <= 2).sum(),
            'Total Sales': len(year_data)
        }
        years_analysis.append(stats)

# Create a DataFrame for better visualization
analysis_df = pd.DataFrame(years_analysis)

import ace_tools as tools; tools.display_dataframe_to_user(name="Quantitative Analysis of Sale Durations by Year", dataframe=analysis_df)
