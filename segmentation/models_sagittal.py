import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet34(weights="DEFAULT")

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
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
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.LeakyReLU(0.021),
            nn.Conv2d(in_channels // 8, in_channels // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
        )
        self.lastconv = nn.Conv2d(in_channels // 8, 5, 1, 1, 0)

    def forward(self, x):
        x = self.inconv(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = F.interpolate(x, size = (384, 384), mode="bilinear", align_corners=True)
        x = self.lastconv(x)
        return x

class SmallUpsampler(nn.Module):

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
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(0.021),
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
        self.upper = Upsampler(in_channels = 512)

    def forward(self, images):
        encoded = self.image_encoder(images)
        mask = self.upper(encoded)
        return mask.sigmoid()

class LumbarPositionModelSagittal(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.position_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Sigmoid()
        )

    def forward(self, images):
        encoded = self.image_encoder(images)
        encoded = self.pooler(encoded)
        encoded = encoded.view(encoded.size(0), -1)
        positions = self.position_decoder(encoded)
        return positions

class LumbarPositionAdnSegmentationModelSagittal(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.upsampler = SmallUpsampler(in_channels=512)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.position_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Sigmoid()
        )

    def forward(self, images):
        encoded = self.image_encoder(images)
        
        mask = self.upsampler(encoded)
        
        pos = self.pooler(encoded)
        pos = pos.view(encoded.size(0), -1)
        pos = self.position_decoder(pos)
        return pos, mask.sigmoid()
