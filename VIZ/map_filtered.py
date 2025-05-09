import pandas as pd
import folium

# Load the dataset with raw string
df = pd.read_csv(r"C:\.....\attempts_locs_20250220.csv")

# Filter for rows where Year is 2024 or 2025 and Subnational Code is US-TN
# filtered = df[(df["Year"].isin([2024, 2025])) & (df["Subnational Code"] == "US-TN")]
# all yrs
filtered = df[df["Subnational Code"] == "US-TN"]

# Export to a new CSV file
output_file = "attempts_TN.csv"
filtered.to_csv(output_file, index=False)

# Confirmation statement
print(f"The file '{output_file}' has been successfully created with {len(filtered)} records.")



# Load the filtered dataset
# input_file = "attempts_TN.csv"
input_file = "attempts_TN_filtered.csv"
df = pd.read_csv(input_file)

# Extract GPS coordinates and drop duplicates
gps_data = df[["Latitude", "Longitude"]].drop_duplicates().dropna()

# Ensure data types are numeric for mapping
gps_data["Latitude"] = pd.to_numeric(gps_data["Latitude"], errors="coerce")
gps_data["Longitude"] = pd.to_numeric(gps_data["Longitude"], errors="coerce")

# Drop rows with invalid coordinates
gps_data = gps_data.dropna()

# Create a map centered at the average location
map_center = [gps_data["Latitude"].mean(), gps_data["Longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=10)

# Add unique markers to the map
for idx, row in gps_data.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"Lat: {row['Latitude']}, Lon: {row['Longitude']}"
    ).add_to(m)

# Save map to HTML file
output_map = "TN_GPS_Map_Unique.html"
m.save(output_map)

# Confirmation statement
print(f"The map '{output_map}' has been successfully created with {len(gps_data)} unique GPS points.")v
