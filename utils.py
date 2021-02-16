def size_str2tuple(_str):
    out = [int(k) for k in _str.split("x")]
    assert len(out) == 2, "Unknown {}. The size should have been [width]x[height].".format(_str)
    return tuple(out)
