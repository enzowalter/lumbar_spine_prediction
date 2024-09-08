import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SagittalImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.squeezenet1_0(weights="DEFAULT")

    def forward(self, x):
        x = self.backbone.features(x)
        x = torch.mean(x, dim = (2, 3))
        return x

class SagittalSliceSelecterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = SagittalImageEncoder()
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, 5)

    def forward(self, images):
        _, s, _, _, _ = images.size()
        encoded = torch.stack([self.image_encoder(images[:, i]) for i in range(s)], dim=1)
        lstm_out, _ = self.lstm(encoded)
        out = self.classifier(lstm_out)
        out = out.permute(0, 2, 1)
        return out.sigmoid()
