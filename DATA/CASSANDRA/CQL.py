# "embedding" of unstructured data
points = {
    "point1": {"x": 12, "y": 10},
    "point2": {"x": 8: "y": 3},
    # ... and so on ...
}

# vs "encoding" of unstructured data

points = {
    "point1::": "dict"
    "point1::x": 12,
    "point1::y": 10,
    "point2::": "dict"
    "point2::x": 8,
    "point2::y": 3,
    # ... and so on ...
}
