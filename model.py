import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import ceil, floor
from typing import Optional, List, Tuple, Union


class LocallyConnected2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_h: int,
        input_w: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
    ) -> None:
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.output_h = floor(
            (input_h + 2 * padding - (kernel_size - 1) - 1) / stride + 1
        )
        self.output_w = floor(
            (input_w + 2 * padding - (kernel_size - 1) - 1) / stride + 1
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(
                self.out_channels,
                self.output_h,
                self.output_w,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )

        self.bias = nn.Parameter(
            torch.randn(1, self.out_channels, self.output_h, self.output_w)
        )

        nn.init.kaiming_normal_(
            self.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, (self.padding,) * 4)

        patches = x_padded.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        out = torch.einsum("ohwcik,nhwcik->nohw", self.weight, patches)
        out = out + self.bias

        return out


class ConvModule(nn.Module):
    def __init__(
        self, in_channels: int, module_config: List[Union[List, Tuple]]
    ) -> None:
        super(ConvModule, self).__init__()

        self.layers = []
        current_channels = in_channels
        for sm_config in module_config:
            if isinstance(sm_config, tuple):
                current_channels = self._add_layer(current_channels, sm_config)
            elif isinstance(sm_config, list):
                sm_layers, r = sm_config
                for _ in range(r):
                    for layer_config in sm_layers:
                        current_channels = self._add_layer(
                            current_channels, layer_config
                        )
            else:
                raise ValueError(f"Wrong module config: {sm_config}")
        self.out_channels = current_channels
        self.layers = nn.Sequential(*self.layers)

    def _add_layer(self, in_channels: int, layer_config: Tuple) -> int:
        layer_type = layer_config[0]

        if layer_type == "c":
            kernel_size, out_channels = layer_config[1:3]
            stride = layer_config[3] if len(layer_config) > 3 else 1

            padding = (
                ceil((kernel_size - stride) / 2)
                if kernel_size >= stride
                else (kernel_size - 1) // 2
            )

            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )

            nn.init.kaiming_normal_(
                layer[0].weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
            )
            self.layers.append(layer)
            in_channels = out_channels

        elif layer_type == "p":
            kernel_size, stride = layer_config[1:]

            self.layers.append(nn.MaxPool2d(kernel_size, stride, ceil_mode=False))

        else:
            raise ValueError(f"Wrong layer type: {layer_type}")

        return in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class YOLOv1(nn.Module):
    custom_backbone_config = [
        [("c", 7, 64, 2), ("p", 2, 2)],
        [("c", 3, 192), ("p", 2, 2)],
        [("c", 1, 128), ("c", 3, 256), ("c", 1, 256), ("c", 3, 512), ("p", 2, 2)],
        [
            [[("c", 1, 256), ("c", 3, 512)], 4],
            ("c", 1, 512),
            ("c", 3, 1024),
            ("p", 2, 2),
        ],
        [[[("c", 1, 512), ("c", 3, 1024)], 2]],
    ]

    detection_head_conv_config = [
        [("c", 3, 1024), ("c", 3, 1024, 2)],
        [("c", 3, 1024), ("c", 3, 1024)],
    ]

    def __init__(self, S: int, B: int, C: int, backbone_type: str = "custom") -> None:
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.backbone_type = backbone_type

        if backbone_type == "custom":
            backbones_modules_list = []
            in_channels = 3
            for module_config in self.custom_backbone_config:
                cm = ConvModule(in_channels, module_config)
                backbones_modules_list.append(cm)
                in_channels = cm.out_channels
            self.backbone = nn.Sequential(*backbones_modules_list)
            backbone_out_channels = 1024
            final_feature_map_size = 14

        elif backbone_type == "resnet18":
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet18.children())[:-2])
            backbone_out_channels = 512
            final_feature_map_size = 14
        else:
            raise ValueError(
                f"Wrong backbone type: {backbone_type}. Available: 'custom', 'resnet18'"
            )

        head_conv_modules_list = []

        in_channels_head = backbone_out_channels
        for i, module_config in enumerate(self.detection_head_conv_config):
            cm = ConvModule(in_channels_head, module_config)
            head_conv_modules_list.append(cm)
            in_channels_head = cm.out_channels

        detection_conv_part = nn.Sequential(*head_conv_modules_list)
        head_conv_out_channels = in_channels_head
        head_final_feature_map_size = final_feature_map_size // 2
        fc_in_channels = head_conv_out_channels

        lc_out_channels = 256
        lc_kernel_size = 3
        lc_stride = 1
        lc_padding = 1

        detection_fc_part = nn.Sequential(
            LocallyConnected2d(
                fc_in_channels,
                lc_out_channels,
                input_h=head_final_feature_map_size,
                input_w=head_final_feature_map_size,
                kernel_size=lc_kernel_size,
                stride=lc_stride,
                padding=lc_padding,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(
                lc_out_channels
                * head_final_feature_map_size
                * head_final_feature_map_size,
                S * S * (self.C + self.B * 5),
            ),
        )

        nn.init.normal_(detection_fc_part[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(detection_fc_part[-1].bias)

        self.detection_head = nn.Sequential(detection_conv_part, detection_fc_part)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        predictions = self.detection_head(features)

        output = predictions.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return output
