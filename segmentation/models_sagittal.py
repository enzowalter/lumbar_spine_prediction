import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ImageEncoder(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b0(weights="DEFAULT")

    def forward(self, x):
        x = self.backbone.features(x)
        return x

class Upsampler(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.021),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(0.021),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )
        self.lastconv = nn.Conv2d(in_channels // 4, 5, 1, 1, 0)

    def forward(self, x):
        x = self.inconv(x)
        x = self.up1(x)
        x = self.up2(x)
        x = F.interpolate(x, size = (96, 96), mode="bilinear", align_corners=True)
        x = self.lastconv(x)
        return x

class LumbarSegmentationModelSagittal(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.upper = Upsampler(in_channels = 1280)

    def forward(self, images):
        encoded = self.image_encoder(images)
        mask = self.upper(encoded)
        return mask.sigmoid()
