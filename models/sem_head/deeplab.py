import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN


class DeeplabV3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=256,
                 out_stride=16,
                 norm_act=nn.BatchNorm2d,
                 pooling=True,
                 pooling_size=None,
                 last_relu=True):
        super(DeeplabV3, self).__init__()

        self.pooling_size = pooling_size

        if out_stride == 16:
            dilations = [6, 12, 18]
        elif out_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[0], padding=dilations[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[1], padding=dilations[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[2], padding=dilations[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)
        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)

        self.pooling = pooling

        self.last_relu = last_relu
        if last_relu:
            self.red_bn = norm_act(out_channels)

        if isinstance(self.map_bn, ABN):
            print("reset norm!")
            self.reset_parameters(self.map_bn.activation, self.map_bn.activation_param)

    def reset_parameters(self, activation, slope):
        gain = nn.init.calculate_gain(activation, slope)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)

        # Global pooling
        if not self.pooling:
            self.global_pooling_bn.eval()

        pool = self._global_pooling(x)  # if training is global avg pooling 1x1, else use pool size
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))
        out += pool

        if self.last_relu:
            out = self.red_bn(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            # this is like Adaptive Average Pooling (1,1)
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode="replicate")
        return pool


class DeeplabV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_stride=16,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None,
                 last_relu=False):
        super(DeeplabV2, self).__init__()
        self.pooling_size = pooling_size

        if out_stride == 16:
            dilations = [1, 6, 12, 18]
        elif out_stride == 8:
            dilations = [6, 12, 18, 24]
        else:
            raise NotImplementedError

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, bias=False, dilation=dilations[0], padding=dilations[0]),
            nn.Conv2d(in_channels, out_channels, 3, bias=False, dilation=dilations[1], padding=dilations[1]),
            nn.Conv2d(in_channels, out_channels, 3, bias=False, dilation=dilations[2], padding=dilations[2]),
            nn.Conv2d(in_channels, out_channels, 3, bias=False, dilation=dilations[3], padding=dilations[3])
        ])

        self.last_relu = last_relu
        if last_relu:
            self.red_bn = norm_act(out_channels)

    def forward(self, x):
        # Map convolutions
        out = sum([m(x) for m in self.map_convs])

        if self.last_relu:
            out = self.red_bn(out)
        return out


def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list
