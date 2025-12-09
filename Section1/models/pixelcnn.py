import torch
from torch import nn
import torch.nn.functional as F

### WARNING: DO NOT EDIT THE CLASS NAME, INITIALIZER, AND GIVEN INPUTS AND ATTRIBUTES. OTHERWISE, YOUR TEST CASES CAN FAIL. ###


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.mask_type = mask_type
        #self.register_buffer("mask", torch.ones_like(self.weight)) NOTE editing initializer
        #self._build_mask()
        self.register_buffer("mask", self._build_mask())

    def _build_mask(self):
        c_out, c_in, k_h, k_w = self.weight.shape #NOTE
        center_h, center_w = k_h // 2, k_w // 2 #NOTE
        
        mask = torch.ones_like(self.weight)

        # Mask type A blocks the current pixel in addition to future pixels; type B allows the current pixel.
        mask[:, :, center_h+1:, :] = 0.0  # NOTE: zero out rows strictly below the current pixel
        mask[:, :, center_h, center_w+1:] = 0.0  # NOTE: zero out columns to the right of the current pixel (respecting mask type)
        if self.mask_type == "A":
            mask[:, :, center_h, center_w] = 0.0  # NOTE: zero the current pixel for type A masks
        return mask

    def forward(self, x):
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResidualConnection(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class PixelCNN(nn.Module):
    def __init__(self, image_channels=3, hidden_channels=64, residual_layers=5, bins=256):
        super().__init__()
        self.conv_cls = MaskedConv2d
        self.image_channels = image_channels
        self.bins = bins
        self.net = self._build_network(image_channels, hidden_channels, residual_layers, bins)

    def _residual_block(self, channels):
        block = ResidualConnection(
            nn.Sequential(
                nn.ReLU(),
                self.conv_cls(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    mask_type="B",
                    bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
        )
        return block

    def _build_network(self, image_channels, hidden_channels, residual_layers, bins):
        layers = [
            self.conv_cls(
                image_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                mask_type="A",
                bias=True,
            ),
            nn.ReLU(),
        ]
        for _ in range(residual_layers):
            layers.append(self._residual_block(hidden_channels))
        layers.extend(
            [
                nn.ReLU(),
                self.conv_cls(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=1,
                    padding=0,
                    mask_type="B",
                    bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, image_channels * bins, kernel_size=1),
            ]
        )
        network = nn.Sequential(*layers)
        return network

    def forward(self, batch):
        x = batch["images"]
        logits = self.net(x)  # NOTE: compute logits over discrete bins
        b, c_bins, h, w = logits.shape
        logits = logits.view(b, self.image_channels, self.bins, h, w)
        targets = batch.get("targets")
        if targets is None:
            targets = (x * (self.bins - 1)).long()  # NOTE: quantise the input images to bin indices, [0, 1] -> [0, bins-1] and convert to long
        else:
            targets = targets.long()
        logits_flat = logits.permute(0, 3, 4, 1, 2).reshape(-1, self.bins)
        targets_flat = targets.permute(0, 2, 3, 1).reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)  # NOTE: compute the cross-entropy loss
        logits = logits.view(b, self.image_channels * self.bins, h, w)
        return {
            "loss": loss,
            "logits": logits,
            "targets": targets,
        }