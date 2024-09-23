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
        if hidden_size is not None:
            self.model = self.modify_classifier()

    def modify_classifier(self):
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
            )
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
            )
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.fc.in_features if hasattr(self.model.head, 'fc') else self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, self.hidden_size),
            )
        else:
            raise NotImplementedError("Unknown classifier structure")
        return self.model

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.backbone = DynamicModelLoader("cspresnet50.ra_in1k", hidden_size=None).model
        #self.backbone = torchvision.models.mnasnet0_5(weights="DEFAULT")

        self.model = timm.create_model(
                        'convnext_small.in12k_ft_in1k',
                        pretrained=True,
        )
        #self.encoder = CustomEncoder()

        # squeezenet 1 0 => {1: 0.5959390862944163, 3: 0.9756345177664975, 5: 0.9908629441624366, 10: 1.0}
        # squeezenet 1 1 => {1: 0.6131979695431472, 3: 0.9736040609137055, 5: 0.9868020304568528, 10: 0.9989847715736041}
        # efficientnet_b0 => overfit
        # shufflenet_v2_x0_5 => overfit
        # mnasnet0 5 => overfit
        # convnext tiny =>  {1: 0.6192893401015228, 3: 0.9766497461928934, 5: 0.9908629441624366, 10: 1.0}
        # BIT tiny => {1: 0.6030456852791878, 3: 0.9776649746192894, 5: 0.9878172588832488, 10: 0.9979695431472081}
        # custom => overfit

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
