def ft_write(d, data):
    s = str(bytearray(data)) if sys.version_info<(3,) else bytes(data)
    return d.write(s)
 
ft_write(d, [OP]*4 + [0])
