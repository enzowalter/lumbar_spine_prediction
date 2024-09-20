import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CropSelecter(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.squeezenet1_0(weights="DEFAULT")
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        features = self.pooler(features)
        score = self.classifier(features)
        return score


