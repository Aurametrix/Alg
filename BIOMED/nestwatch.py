import pandas as pd

# Load the dataset
input_file = "attempts_locs_20250220.csv"

# Load the dataset
df = pd.read_csv(input_file)


# Ensure relevant columns are in the correct data type
df["First Lay Date"] = pd.to_datetime(df["First Lay Date"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

# Drop rows with missing data in relevant columns
df = df.dropna(subset=["First Lay Date", "Latitude", "Longitude", "Species Name"])

# Extract year, month, and day
df["Year"] = df["First Lay Date"].dt.year
df["Month_Day"] = df["First Lay Date"].dt.strftime("%m-%d")

# Extract the earliest month/day for each species in each year
earliest_dates = df.loc[df.groupby(["Species Name", "Year"])["First Lay Date"].idxmin()]

# Print the results
for _, row in earliest_dates.iterrows():
    print(f"Species: {row['Species Name']} - Year: {row['Year']} - Earliest Date: {row['Month_Day']}")
    print(f"Latitude: {row['Latitude']}, Longitude: {row['Longitude']}\n")
