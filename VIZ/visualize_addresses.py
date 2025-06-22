"""
visualize_addresses.py
----------------------
Create an interactive HTML map with one marker per address.
The script uses OpenStreetMap/Nominatim (free) for geocoding and Folium
for rendering.  It’s written for clarity—feel free to refactor!
"""
import sys
from pathlib import Path
from statistics import mean

import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def geocode_addresses(addresses, user_agent: str = "addr-mapper") -> list[tuple[str, float, float]]:
    """
    Geocode each address with Nominatim and return
    [(address, lat, lon), ...].  Unresolved addresses are skipped.
    """
    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    # RateLimiter: be polite (OSM’s policy = 1 req/sec)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    results = []
    for addr in addresses:
        location = geocode(addr)
        if location is None:
            print(f"[WARN] Could not geocode: {addr}", file=sys.stderr)
            continue
        results.append((addr, location.latitude, location.longitude))
    return results


def make_map(locations, out_html="address_map.html"):
    """
    locations: iterable of (label, lat, lon)
    Saves an interactive HTML map to *out_html*.
    """
    if not locations:
        raise ValueError("No locations to plot!")

    # Center map on the average lat/lon
    avg_lat = mean(lat for _, lat, _ in locations)
    avg_lon = mean(lon for _, _, lon in locations)

    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles="OpenStreetMap")

    for label, lat, lon in locations:
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(label, show=True, sticky=True),
            tooltip=label,
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(fmap)

    fmap.save(out_html)
    print(f"✅  Map saved to {Path(out_html).resolve()}")


if __name__ == "__main__":
    # Example: All addresses in Vonore, TN (you can also pass your own list or read from a file)
    ADDRESSES = [
        "101 First Trail, Town, TN",
        "133 Second Trail, Town, TN",

    ]

    # If the user passed addresses on the command line, override the default list
    if len(sys.argv) > 1:
        ADDRESSES = [" ".join(sys.argv[1:])] if len(sys.argv) == 2 else sys.argv[1:]

    locations = geocode_addresses(ADDRESSES)
    make_map(locations)
