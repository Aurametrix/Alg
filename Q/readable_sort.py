def length_and_alphabetical(string):
    """Return sort key: length first, then case-normalized string."""
    return (len(string), string.casefold())

colors = ["Goldenrod", "Purple", "Salmon", "Turquoise", "Cyan"])
colors_by_length = sorted(colors, key=length_and_alphabetical)
