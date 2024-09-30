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
        # self.backbone = DynamicModelLoader('convnext_base.fb_in22k_ft_in1k', hidden_size=features_size).model
        self.backbone = timm.create_model("focalnet_small_lrf.ms_in1k", pretrained=True)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.backbone.forward_features(x)
        # features = torch.mean(features, dim = (2, 3))
        features = self.pooler(features)
        features = features.flatten(start_dim = 1)
        return features

class CropSelecter(nn.Module):

    def __init__(self):
        super().__init__()
        self.features_size = 768
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

class SqueezeNetImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.squeezenet1_0(weights="DEFAULT")

    def forward(self, x):
        x = self.model.features(x)
        features = torch.mean(x, dim = (2, 3))
        features = features.flatten(start_dim=1)
        return features


class REM_CropSelecterModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([SqueezeNetImageEncoder() for _ in range(3)])
        self.lstm = nn.LSTM(512, 512 // 4, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 // 2, 1),
        )

    def forward(self, images, mode="inference"):
        _, s, _, _, _ = images.size()
        if mode == "train":

            r_b = torch.randint(0, len(self.encoders), (1,)).item()
            encoder = self.encoders[r_b]

            encoded = torch.stack([encoder(images[:, i]) for i in range(s)], dim=1)
            lstm_out, _ = self.lstm(encoded)
            out = self.classifier(lstm_out)

            return out.sigmoid().squeeze(-1)

        else:
            encoded = list()
            for encoder in self.encoders:
                _encoded = torch.stack([encoder(images[:, i]) for i in range(s)], dim=1)
                encoded.append(_encoded)
            encoded = torch.stack(encoded, dim = 1)
            encoded = torch.mean(encoded, dim = 1)
            lstm_out = self.lstm(encoded)[0]
            out = self.classifier(lstm_out)
            return out.sigmoid().squeeze(-1)

class REM_CropSelecterModel_Scripted(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([SqueezeNetImageEncoder() for _ in range(3)])
        self.lstm = nn.LSTM(512, 512 // 4, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 // 2, 1),
        )
    
    def forward(self, images):
        _, s, _, _, _ = images.size()
        encoded = list()
        for encoder in self.encoders:
            _encoded = torch.stack([encoder(images[:, i]) for i in range(s)], dim=1)
            encoded.append(_encoded)
        encoded = torch.stack(encoded, dim = 1)
        encoded = torch.mean(encoded, dim = 1)
        lstm_out = self.lstm(encoded)[0]
        out = self.classifier(lstm_out)
        return out.sigmoid().squeeze(-1)