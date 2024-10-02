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

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        return x1, x2, x3, x4

class UpBlock(nn.Module):

    def __init__(self, ich, och):
        super().__init__()

        self.inconv = nn.Sequential(
            nn.Conv2d(ich, och, 3, 1, 1, bias=False),
            nn.BatchNorm2d(och),
            nn.ReLU(),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(och * 2, och, 3, 1, 1, bias=False),
            nn.BatchNorm2d(och),
            nn.ReLU(),
        )
    
    def forward(self, x1, x2):
        b, c, h, w = x1.size()
        x1 = self.inconv(x1)
        x1 = F.interpolate(x1, size = (w*2, h*2), mode="bilinear", align_corners=True)

        x = torch.cat([x1, x2], dim = 1)
        x = self.outconv(x)
        return x

class PositionHead(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(features_size, features_size // 2),
            nn.ReLU(),
            nn.Linear(features_size // 2, 10),
        )

    def forward(self, x):
        x = self.pooler(x)
        x = x.flatten(start_dim = 1)
        x = self.fc(x)
        return x

class Upsampler(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)

        self.lastconv = nn.Conv2d(64, 5, 1, 1, 0)

    def forward(self, x1, x2, x3, x4):
        u = self.inconv(x4)
        u = self.up1(u, x3)
        u = self.up2(u, x2)
        u = self.up3(u, x1)
        u = F.interpolate(u, size = (384, 384), mode="bilinear", align_corners=True)
        u = self.lastconv(u)
        return u

class LumbarSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.upper_head = Upsampler(in_channels = 512)
        self.position_head = PositionHead(features_size=512)

    def forward(self, images):
        x1, x2, x3, x4 = self.image_encoder(images)
        mask = self.upper_head(x1, x2, x3, x4)
        positions = self.position_head(x4)
        return mask.sigmoid(), positions.sigmoid()
