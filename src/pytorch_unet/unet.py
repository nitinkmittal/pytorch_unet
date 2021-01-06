from typing import Tuple

import torch
import torch.nn as nn


def conv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_class: int):
        super().__init__()  # to call __init__ of parent class (nn.Module)

        self.layer_0_conv2d = conv2d_block(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.layer_1_conv2d = conv2d_block(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.layer_2_conv2d = conv2d_block(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.layer_3_conv2d = conv2d_block(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2))

        self.upsample2d = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.layer_4_inv_conv2d = conv2d_block(
            in_channels=256 + 512,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.layer_5_inv_conv2d = conv2d_block(
            in_channels=128 + 256,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.layer_6_inv_conv2d = conv2d_block(
            in_channels=64 + 128,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.layer_7_conv2d = nn.Conv2d(
            in_channels=64,
            out_channels=n_class,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

    def forward(self, input):

        # ? * C * H * W -> ? * 64 * H * W
        layer_0_conv2d = self.layer_0_conv2d(input)
        # ? * 64 * H * W -> ? * 64 * H/2 * W/2
        layer_0_maxpool2d = self.maxpool2d(layer_0_conv2d)

        # ? * 64 * H/2 * W/2 -> ? * 128 * H/2 * W/2
        layer_1_conv2d = self.layer_1_conv2d(layer_0_maxpool2d)
        # ? * 128 * H/2 * W/2  -> ? * 128 * H/4 * W/4
        layer_1_maxpool2d = self.maxpool2d(layer_1_conv2d)

        # ? * 128 * H/4 * W/4 -> ? * 256 * H/4 * W/4
        layer_2_conv2d = self.layer_2_conv2d(layer_1_maxpool2d)
        # ? * 256 * H/4 * W/4  -> ? * 256 * H/8 * W/8
        layer_2_maxpool2d = self.maxpool2d(layer_2_conv2d)

        # ? * 256 * H/8 * W/8 -> ? * 512 * H/8 * W/8
        layer_3_conv2d = self.layer_3_conv2d(layer_2_maxpool2d)

        # ? * 512 * H/8 * W/8 -> ? * 512 * H/4 * W/4
        layer_4_upsample2d = self.upsample2d(layer_3_conv2d)
        # ? * (256 + 512) * H/4 * W/4 -> ? * 256 * H/4 * W/4
        layer_4_inv_conv2d = self.layer_4_inv_conv2d(
            torch.cat([layer_2_conv2d, layer_4_upsample2d], dim=1)
        )

        # ? * 256 * H/4 * W/4 -> ? * 256 * H/2 * W/2
        layer_5_upsample2d = self.upsample2d(layer_4_inv_conv2d)
        # ? * (128 + 256) * H/2 * W/2 -> ? * 128 * H/2 * W/2
        layer_5_inv_conv2d = self.layer_5_inv_conv2d(
            torch.cat([layer_1_conv2d, layer_5_upsample2d], dim=1)
        )

        # ? * 128 * H/2 * W/2 -> ? * 128 * H * W
        layer_6_upsample2d = self.upsample2d(layer_5_inv_conv2d)
        # ? * (64 + 128) * H * W -> ? * 64 * H * W
        layer_6_inv_conv2d = self.layer_6_inv_conv2d(
            torch.cat([layer_0_conv2d, layer_6_upsample2d], dim=1)
        )
        # ? * 64 * H * W -> ? * n_class * H * W
        layer_7_conv2d = self.layer_7_conv2d(layer_6_inv_conv2d)

        return layer_7_conv2d
