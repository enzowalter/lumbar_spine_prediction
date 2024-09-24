import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
                        'convnext_small.in12k_ft_in1k',
                        pretrained=True,
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        features = torch.mean(x, dim = (2, 3))
        return features

class SagittalSliceSelecterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.lstm = nn.LSTM(768, 768 // 4, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(768 // 2, 5),
        )

    def forward(self, images):
        _, s, _, _, _ = images.size()
        encoded = torch.stack([self.image_encoder(images[:, i]) for i in range(s)], dim=1)
        lstm_out, _ = self.lstm(encoded)
        out = self.classifier(lstm_out)
        out = out.permute(0, 2, 1)
        return out.sigmoid()
