import torch.nn as nn
from networks.components import get_block, get_sblock, get_module
from utils import *

def get_model(_name):
    return {
        "net": Net,
    }[_name]

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.clen = args.g_coder_length
        self.mlen = args.g_mid_length

        self.layer_1 = get_sblock("ConvBlock")(args.g_in_channels, args.g_n_channels, kernel_size=5, stride=1, groups=1, length=1, act=args.g_act)
        self.layer_2 = get_sblock("RECABlock")(args.g_n_channels, args.g_n_channels, kernel_size=3, stride=1, groups=1, length=1, act=args.g_act)

        for i in range(self.clen):
            setattr(self,"enc_%d" % (i+1), get_block("U")(
                args.g_n_channels*2**(i), args.g_n_channels*2**(i+1),
                block_names=args.g_bns,
                subsampling=args.g_downsampler,
                act=args.g_act
                ))

        for i in range(self.mlen):
            setattr(self, "mid_%d" % (i+1), get_sblock("ResBlock")(
                args.g_n_channels*2**self.clen, args.g_n_channels*2**self.clen, length=2,
                kernel_size=3, stride=1, act=args.g_act
                ))

        for i in range(self.clen):
            setattr(self,"dec_%d" % (i+1), get_block("U")(
                args.g_n_channels*2**(self.clen-i), args.g_n_channels*2**(self.clen-i-1),
                block_names=args.g_bns,
                subsampling=args.g_upsampler,
                act=args.g_act
                ))

        self.layer_3 = get_sblock("ConvBlock")(args.g_n_channels, args.g_n_channels, kernel_size=3, stride=1, groups=1, length=1, act=args.g_act)
        self.final_layer = nn.Sequential(
            get_module("Conv2d")(args.g_n_channels, 3, kernel_size=3, stride=1, padding=1, groups=1, bias=True),
            nn.Tanh()
        )

    def forward(self, X):
        skips = []

        X = self.layer_1(X)
        X = self.layer_2(X)
        skips.append(X)
        for i in range(self.clen):
            X = getattr(self, "enc_%d" % (i+1))(X)
            skips.append(X)

        out = skips.pop()
        for i in range(self.mlen):
            out = getattr(self, "mid_%d" % (i+1))(out)
        for i in range(self.clen):
            out = getattr(self, "dec_%d" % (i+1))(out, skips[-i-1])

        out = self.layer_3(out)

        return self.final_layer(out)
