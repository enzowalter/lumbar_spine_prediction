import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class REM_Policy(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, 
            nhead=8, 
            dim_feedforward=feature_size // 2, 
            dropout=0.21,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.fc_weights = nn.Sequential(
            nn.Dropout(0.21),
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, 1)
        )

    def forward(self, encodeds):
        x = self.transformer_encoder(encodeds)
        encodeds_weights = self.fc_weights(x)
        crop_probs = F.softmax(encodeds_weights.squeeze(-1), dim=1)
        return crop_probs

class REM_CropsEncoder(nn.Module):

    def __init__(self, backbone_name, unification_size):
        super().__init__()
        self.backbone_name = backbone_name
        self.unification_size = unification_size
        # encoding
        self.model = timm.create_model(self.backbone_name, pretrained=True)
        self.fc_proj = nn.Linear(self.model.num_features, self.unification_size)
        # crop selection
        self.policy = REM_Policy(self.unification_size)

    def _forward(self, crop):
        encoded = self.model.forward_features(crop)
        if len(encoded.shape) == 3: # transformer b s f
            encoded = encoded[:, 0]
        else:
            encoded = torch.mean(encoded, dim = (2, 3))
            encoded = encoded.flatten(start_dim=1)
        encoded = self.fc_proj(encoded)
        return encoded

    def forward(self, crops):
        b, s, c, h, w = crops.size()
        encoded_crops = torch.stack([self._forward(crops[:, i]) for i in range(s)], dim = 1)
        crops_probs = self.policy(encoded_crops)
        selected_crops = torch.argmax(crops_probs, dim=1) 
        features = torch.einsum("bsf, bs -> bf", encoded_crops, crops_probs)
        return features, crops_probs, selected_crops

class REM_Classifier(nn.Module):
    def __init__(self, features_size, nb_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.21),
            nn.Linear(features_size, features_size // 2),
            nn.ReLU(),
            nn.Linear(features_size // 2, nb_classes)
        )
    
    def forward(self, features):
        return self.classifier(features)

class REM(nn.Module):

    def __init__(self, n_classes, backbones, n_classifiers, unification_size):
        super().__init__()

        self.nb_classes = n_classes
        self.nb_classifiers = n_classifiers
        self.nb_encoders = len(backbones)
        self.backbones = backbones
        self.unification_size = unification_size

        self.encoders = nn.ModuleList([
            REM_CropsEncoder(backbone, self.unification_size) for backbone in self.backbones
        ])

        self.classifiers = nn.ModuleList([
            REM_Classifier(self.unification_size, self.nb_classes) for _ in range(self.nb_classifiers)
        ])

        self.weights = nn.Parameter(torch.ones(self.nb_classifiers, self.nb_encoders))

    def forward_encoders(self, crops):
        features = list()
        weights = list()
        selected = list()
        for encoder in self.encoders:
            f, w, s = encoder(crops)
            features.append(f)
            weights.append(w)
            selected.append(s)
        features = torch.stack(features, dim = 1)
        weights = torch.stack(weights, dim = 1)
        selected = torch.stack(selected, dim = 1)
        return features, weights, selected

    def forward(self, crops, mode):
        
        if mode == "train":
            r_b = torch.randint(0, self.nb_encoders, (1,)).item()
            r_c = torch.randint(0, self.nb_classifiers, (1,)).item()
            backbone = self.encoders[r_b]
            classifier = self.classifiers[r_c]
            encoded, weight, selected = backbone(crops)
            output = classifier(encoded)
            return output, weight, selected
    
        if mode == "train_gate":
            final_output = list()
            
            with torch.no_grad():
                _encodeds, weights, selected = self.forward_encoders(crops)
    
            for idx, classifier in enumerate(self.classifiers):
                classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
                classifier_weights = torch.softmax(self.weights[idx], dim=0)
                classifier_output = torch.einsum("bsf,s->bf", classified_, classifier_weights)
                final_output.append(classifier_output)
            final_output = torch.stack(final_output, dim=1)
            final_output = torch.mean(final_output, dim=1)
            return final_output, weights, selected

        if mode == "inference":
            final_output = list()
            _encodeds, weights, selected = self.forward_encoders(crops)
            for idx, classifier in enumerate(self.classifiers):
                classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
                classifier_weights = torch.softmax(self.weights[idx], dim=0)
                classifier_output = torch.einsum("bsf,s->bf", classified_, classifier_weights)
                final_output.append(classifier_output)
            final_output = torch.mean(torch.stack(final_output, dim=1), dim=1)
            return final_output, weights, selected
