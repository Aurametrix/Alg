import torch

def encode(points, padded_length=100):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # Find the extremes so we can scale everything
    # to fit into the [-0.5, 0.5] range
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    
    # I’m scaling y coordinates by the same ratio to keep things
    # proportional (otherwise we might lose some precious information).
    # This computes how much to shift scaled y values by so that
    # they cluster around 0.
    y_shift = ((max_y - min_y) / (max_x - min_x)) / 2.0
    
    # Create the tensor
    input_tensor = torch.zeros([2, padded_length])

    def normalize_x(x):
        return (x - min_x) / (max_x - min_x) - 0.5
    def normalize_y(y):
        return (y - min_y) / (max_x - min_x) - y_shift

    # Fill the tensor with normalized values
    for i in range(min(padded_length, len(points))):
        input_tensor[0][i] = normalize_x(points[i][0] * 1.0)
        input_tensor[1][i] = normalize_y(points[i][1] * 1.0)
        continue
    return input_tensor
