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
            nn.Dropout(0.5),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
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
        encoded_crops = self._forward(crops.view(-1, c, h, w))
        encoded_crops = encoded_crops.view(b, s, -1)

        crops_probs = self.policy(encoded_crops)
        selected_crops = torch.argmax(crops_probs, dim=1) 
        features = torch.einsum("bsf, bs -> bf", encoded_crops, crops_probs)
        return features, crops_probs, selected_crops

class REM_LinearClassifier(nn.Module):
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
            REM_LinearClassifier(self.unification_size, self.nb_classes) for _ in range(self.nb_classifiers)
        ])

        self.weights_encoders = nn.Parameter(torch.ones(self.nb_encoders))

    def forward_encoders(self, crops):
        batch_size = crops.size(0)
        seq_length = crops.size(1)
        features_shape = (batch_size, self.nb_encoders, self.unification_size)
        weights_shape = (batch_size, self.nb_encoders, seq_length)
        selected_shape = (batch_size, self.nb_encoders)

        features = torch.empty(features_shape, device=crops.device)
        weights = torch.empty(weights_shape, device=crops.device)
        selected = torch.empty(selected_shape, device=crops.device)

        for i, encoder in enumerate(self.encoders):
            f, w, s = encoder(crops)
            features[:, i, :] = f
            weights[:, i, :] = w
            selected[:, i] = s

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
    
        if mode == "train_encoder_weights":
            with torch.no_grad():
                _encodeds, weights, selected = self.forward_encoders(crops)
                classified_outputs = torch.stack([classifier(_encodeds) for classifier in self.classifiers], dim=1)
            
            weights_encoder = torch.softmax(self.weights_encoders, dim = 0)
            classified_outputs = torch.einsum("bnec, e -> bnc", classified_outputs, weights_encoder)
            final_output = classified_outputs.mean(dim=1)
            return final_output, weights, selected

        if mode == "inference":
            _encodeds, weights, selected = self.forward_encoders(crops)
            classified_outputs = torch.stack([classifier(_encodeds) for classifier in self.classifiers], dim=1)
            weights_encoder = torch.softmax(self.weights_encoders, dim = 0)
            classified_outputs = torch.einsum("bnec, e -> bnc", classified_outputs, weights_encoder)
            final_output = classified_outputs.mean(dim=1)
            return final_output, weights, selected



class REM_torchscript(nn.Module):

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
            REM_LinearClassifier(self.unification_size, self.nb_classes) for _ in range(self.nb_classifiers)
        ])

        self.weights_encoders = nn.Parameter(torch.ones(self.nb_encoders))

    def forward_encoders(self, crops):
        features = torch.empty((crops.size(0), self.nb_encoders, self.unification_size), device=crops.device, dtype=crops.dtype)
        for i, encoder in enumerate(self.encoders):
            f, _, _ = encoder(crops)
            features[:, i, :] = f
        return features

    def forward(self, crops):
        _encodeds = self.forward_encoders(crops)
        classified_outputs = torch.stack([classifier(_encodeds) for classifier in self.classifiers], dim=1)
        weights_encoder = torch.softmax(self.weights_encoders, dim = 0)
        classified_outputs = torch.einsum("bnec, e -> bnc", classified_outputs, weights_encoder)
        final_output = classified_outputs.mean(dim=1)
        return final_output

