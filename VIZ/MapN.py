import pandas as pd
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from folium import plugins
import os

# Read CSV
df = pd.read_csv("homes.csv")

# Convert longitude to numbers
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Check for bad coordinates
if df["longitude"].isna().any():
    print("WARNING: Some longitude values could not be converted:")
    print(df[df["longitude"].isna()])

# Remove rows with missing coordinates
df = df.dropna(subset=["latitude", "longitude", "neighborhood"])

# Calculate center of map
center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()

print(f"Map center: {center_lat}, {center_lon}")
print(f"Number of homes: {len(df)}")

# Create map with CartoDB Positron (lightweight, always works)
# This is a minimal basemap - perfect for your needs
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles="CartoDB positron",  # Lightweight, minimal detail
    prefer_canvas=True
)

# Optional: Add a second layer with slightly more detail (CartoDB Voyager)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/voyager_labels_under/{z}/{x}/{y}{r}.png",
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    name="Detailed View",
    overlay=False,
    control=True
).add_to(m)

# Get neighborhoods
neighborhoods = df["neighborhood"].unique()

print("Neighborhoods:")
for n in neighborhoods:
    print("  ", n)

# Create colors for all neighborhoods
colors = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]

color_map = {
    neighborhood: colors[i % len(colors)]
    for i, neighborhood in enumerate(neighborhoods)
}

# Add layer groups for each neighborhood (better organization)
layer_groups = {
    neighborhood: folium.FeatureGroup(name=neighborhood, show=True)
    for neighborhood in neighborhoods
}

# Add homes to map
for _, row in df.iterrows():
    neighborhood = row["neighborhood"]
    
    # Create popup with more details
    popup_text = f"""
    <b>{neighborhood}</b><br>
    Lat: {row['latitude']:.4f}<br>
    Lon: {row['longitude']:.4f}
    """
    
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=6,
        color=color_map[neighborhood],
        fill=True,
        fill_color=color_map[neighborhood],
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=250),
        weight=2
    ).add_to(layer_groups[neighborhood])

# Add all layer groups to map
for layer_group in layer_groups.values():
    layer_group.add_to(m)

# Add layer control (toggle neighborhoods on/off)
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Add a title/legend to the map
title_html = '''
             <div style="position: fixed; 
                     top: 10px; left: 50px; width: 250px; height: auto;
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:14px; padding: 10px;
                     border-radius: 5px;
                     box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
             <h3 style="margin-top: 0;">Neighborhood Map</h3>
             <p style="margin: 5px 0;"><b>Total Homes:</b> {}</p>
             <p style="margin: 5px 0;"><b>Neighborhoods:</b> {}</p>
             </div>
             '''.format(len(df), len(neighborhoods))
m.get_root().html.add_child(folium.Element(title_html))

# Save map
m.save("neighborhood_map.html")

print()
print("✓ Map saved as neighborhood_map.html")
print(f"✓ Using lightweight CartoDB basemap (no tile access issues)")
print(f"✓ Toggle neighborhoods on/off using layer control (top right)")
