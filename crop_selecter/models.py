import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CropEncoder(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(576, features_size),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.pooler(features)
        features = features.view(features.size(0), -1)
        features = self.classifier(features)
        return features

class CropSelecter(nn.Module):

    def __init__(self, features_size=256):
        super().__init__()
        self.features_size = features_size
        self.encoder = CropEncoder(self.features_size)
        self.lstm = nn.LSTM(self.features_size, self.features_size // 4, bidirectional=True, batch_first=True, dropout=0.021, num_layers=2)
        self.classifier = nn.Linear(self.features_size // 2, 1)

    def forward(self, crops):
        b, s, c, h, w = crops.size()

        features = torch.empty(b, s, self.features_size).to(crops.device)
        for i in range(s):
            features[:, i] = self.encoder(crops[:, i])
        
        lstm_out, _ = self.lstm(features)
        scores = self.classifier(lstm_out)
        return scores.sigmoid().squeeze(-1)

