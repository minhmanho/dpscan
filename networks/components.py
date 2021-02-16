from torch import nn
from networks.acts import get_act
from networks.blurpool import get_blurpool

def get_block(_name):
    return {
        "U": UBlock,
    }[_name]

def get_subsampler(_name, nchannels):
    return {
        "down_blurmax": nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), get_blurpool("down_2d")(channels=nchannels, filt_size=3, stride=2)),
        "up_blurbilinear": get_blurpool("up")(channels=nchannels),
    }[_name]

class BaseBlock(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        block_names=["ConvBlock-1", "ConvBlock-1"],
        kernels=[3,3],
        subsampling="none",
        act="evo"):
        super(BaseBlock, self).__init__()

        channels = [in_channels]
        channels += [out_channels for _ in range(len(block_names))]

        self.block_names = [k.split('-') for k in block_names]

        for i in range(len(channels)-1):
            setattr(
                self, "block_%d" % (i+1),
                get_sblock(self.block_names[i][0])(
                    channels[i], channels[i+1], length=int(self.block_names[i][1]),
                    kernel_size=kernels[i], act=act
                    ))

    def forward(self, X, _skip_feat=None):
        pass

class UBlock(BaseBlock):
    def __init__(
        self,
        in_channels, out_channels,
        block_names=["ConvBlock-1", "ConvBlock-1"],
        kernels=[3,3],
        subsampling="none",
        act="evo"):
        super().__init__(in_channels, out_channels, block_names, kernels, subsampling, act)
        self.subsampler = get_subsampler(subsampling, in_channels) if subsampling != "none" else None

    def forward(self, X, _skip_feat=None):
        if self.subsampler is not None:
            X = self.subsampler(X)
        X = self.block_1(X)
        if _skip_feat is not None:
            X = X + _skip_feat
        X = self.block_2(X)
        return X

def get_sblock(block_name):
    return {
        "ConvBlock": ConvBlock,
        "ResBlock": ResBlock,
        "RECABlock": RECABlock,
        "RECAUBlock": RECAUBlock,
    }[block_name]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, length=1, act="lrelu"):
        super(ConvBlock, self).__init__()
        self.layers = self.make_layers(in_channels, out_channels, kernel_size, stride, groups, length, act, mn="Conv2d")

    def forward(self, X):
        return self.layers(X)

    @staticmethod
    def make_layers(in_channels, out_channels, kernel_size=3, stride=1, groups=1, length=1, act="lrelu", mn="Conv2d"):
        padding = (kernel_size - 1) // 2

        tmp = [in_channels, out_channels]
        while len(tmp) < length + 1:
            tmp.append(tmp[-1])

        layers = []
        for i in range(length):
            layers += [
                get_module(mn)(tmp[i], tmp[i+1], kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True),
                get_act(act)(tmp[i+1])
            ]
        return nn.Sequential(*layers)

class ResBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, length=2, act="lrelu"):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, length=length, act=act)

    def forward(self, X):
        return X + self.layers(X)

class RECABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, length=1, act="lrelu"):
        super(RECABlock, self).__init__()

        layers = []
        for i in range(2):
            layers.append(get_module('Conv2d')(out_channels, out_channels, kernel_size, padding=1, bias=True))
            layers.append(get_act(act)(out_channels))
        layers.append(get_module('ECA')(out_channels, 3))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = self.layers(x)
        return res + x

class RECAUBlock(RECABlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, length=1, act="lrelu"):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, length=length, act=act)
        self.conv_1 = get_module('Conv2d')(in_channels, out_channels, kernel_size, padding=1, bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        res = self.layers(x)
        return res + x

def get_module(_name):
    return {
        'Conv2d': nn.Conv2d,
        'ECA': ECALayer,
    }[_name]

class ECALayer(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
