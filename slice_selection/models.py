import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class SagittalImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(model_name="tf_efficientnet_lite0.in1k", pretrained=True)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = torch.mean(x, dim = (2, 3))
        return x

class SagittalSliceSelecterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = SagittalImageEncoder()
        self.lstm = nn.LSTM(1280, 256, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
        )

    def forward(self, images):
        _, s, _, _, _ = images.size()
        encoded = torch.stack([self.image_encoder(images[:, i]) for i in range(s)], dim=1)
        lstm_out, _ = self.lstm(encoded)
        out = self.classifier(lstm_out)
        out = out.permute(0, 2, 1)
        return out.sigmoid()
