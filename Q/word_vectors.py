import math
def distance2d(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
import json

color_data = json.loads(open("xkcd.json").read())
The following function converts colors from hex format (#1a2b3c) to a tuple of integers:


def hex_to_int(s):
    s = s.lstrip("#")
    return int(s[:2], 16), int(s[2:4], 16), int(s[4:6], 16)
And the following cell creates a dictionary and populates it with mappings from color names to RGB vectors for each color in the data:


colors = dict()
for item in color_data['colors']:
    colors[item["color"]] = hex_to_int(item["hex"])
