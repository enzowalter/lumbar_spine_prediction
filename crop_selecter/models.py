import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class DynamicModelLoader:
    def __init__(self, model_name, pretrained=True, hidden_size=256):
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.model = self.modify_classifier()

    def modify_classifier(self):
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.fc.in_features if hasattr(self.model.head, 'fc') else self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError("Unknown classifier structure")
        return self.model


class CropEncoder(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.backbone = DynamicModelLoader('convnext_base.fb_in22k_ft_in1k', hidden_size=features_size).model

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = torch.mean(features, dim = (2, 3))
        return features

class CropSelecter(nn.Module):

    def __init__(self, features_size=256):
        super().__init__()
        self.features_size = 1024
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

