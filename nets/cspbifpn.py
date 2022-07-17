from collections import OrderedDict
import torch.nn as nn
from nets.impovebifpn import BiFpn
from nets.CSPdarknet import darknet53


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        # 2048-1024-2048-1024
        # ([512, 1024], 1024) 1024-512-1024-512
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


# spp
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class CspBifpn(nn.Module):
    def __init__(self, anchors_mask, num_classes, num_channels=512, conv_channels=[256, 512, 1024]):
        super(CspBifpn, self).__init__()
        # num_channels in order to make all channel be same to integrate features
        self.num_channels = num_channels
        self.conv_channels = conv_channels

        self.backbone = darknet53(False)
        self.conv1 = make_three_conv([512, 1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)
        self.conv3 = conv2d(512, 1024, 1)
        self.bifpn = nn.Sequential(
            *[BiFpn(self.num_channels,
                    conv_channels,
                    True if _ == 0 else False,
                    attention=True,
                    use_p8=False)
              for _ in range(3)])

        self.yolo_head3 = yolo_head([2 * num_channels, len(anchors_mask[0]) * (5 + num_classes)], num_channels)
        self.yolo_head2 = yolo_head([2 * num_channels, len(anchors_mask[1]) * (5 + num_classes)], num_channels)
        self.yolo_head1 = yolo_head([2 * num_channels, len(anchors_mask[2]) * (5 + num_classes)], num_channels)

    def forward(self, x):
        p1, p2, p3, p4, p5 = self.backbone(x)
        p5 = self.conv1(p5)
        p5 = self.SPP(p5)
        p5 = self.conv2(p5)
        p5 = self.conv3(p5)
        feature = [p1, p2, p3, p4, p5]
        feature = self.bifpn(feature)  # feature = [p1, p2, p3, p4, p5]
        p1, p2, p3, p4, p5 = feature

        out2 = self.yolo_head3(p3)
        out1 = self.yolo_head2(p4)
        out0 = self.yolo_head1(p5)

        return out0, out1, out2

