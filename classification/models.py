import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class REM_ModelLoader:
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
            )
        else:
            raise NotImplementedError("Unknown classifier structure")
        return self.model

class REM_Encoder(nn.Module):
    def __init__(self, features_size, backbone_name):
        super().__init__()
        self.name = backbone_name
        self.model = REM_ModelLoader(model_name=backbone_name, hidden_size=features_size).model
    def forward(self, x):
        return self.model(x)

class REM_Classifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, n_classes)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

class REM(nn.Module):
    def __init__(self, backbones, n_fold_classifier, features_size, n_classes):
        super().__init__()
        self.nb_encoders = len(backbones)
        self.nb_classifiers = n_fold_classifier
        self.features_size = features_size
        self.n_classes = n_classes
        
        self.encoders = nn.ModuleList([
            REM_Encoder(features_size, backbone) for backbone in backbones
        ])
        self.classifiers = nn.ModuleList([
            REM_Classifier(features_size, n_classes) for _ in range(n_fold_classifier)
        ])
        self.weights_encoders = nn.Parameter(torch.ones(len(self.encoders)))

    def forward_encoders(self, crop):
        encodeds = torch.stack([encoder(crop) for encoder in self.encoders], dim=1)
        return encodeds

    def forward(self, crop, mode):
        
        if mode == "train":
            r_b = torch.randint(0, self.nb_encoders, (1,)).item()
            r_c = torch.randint(0, self.nb_classifiers, (1,)).item()
            backbone = self.encoders[r_b]
            classifier = self.classifiers[r_c]
            encoded = backbone(crop)
            output = classifier(encoded)
            return output

        if mode == "inference":
            final_output = list()
            _encodeds = self.forward_encoders(crop)
            for classifier in self.classifiers:
                classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
                classifier_output = torch.einsum("bsf,s->bf", classified_, torch.softmax(self.weights_encoders, dim=0))
                final_output.append(classifier_output)
            final_output = torch.stack(final_output, dim=1)
            final_output = torch.mean(final_output, dim=1)
            return final_output
