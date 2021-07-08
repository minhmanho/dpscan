import glob
import os
import os.path as osp

_EXTS = ["png", "jpg", "jpeg", "bmp"]

def size_str2tuple(_str):
    out = [int(k) for k in _str.split("x")]
    assert len(out) == 2, "Unknown {}. The size should have been [width]x[height].".format(_str)
    return tuple(out)

def read_images(_dir):
    img_names = [glob.glob(osp.join(_dir, '*.'+_ext)) for _ext in _EXTS]
    return [osp.basename(k) for sublist in img_names for k in sublist]
