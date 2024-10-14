import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import time

############################################################################################
############################################################################################
# WEIGHTING
############################################################################################
############################################################################################

class REM_CropWeighter(nn.Module):
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
            nn.Linear(feature_size // 2, 1),
        )

    def forward(self, encodeds):
        x = self.transformer_encoder(encodeds)
        encodeds_weights = self.fc_weights(x)
        crop_weights = F.softmax(encodeds_weights.squeeze(-1), dim=1)
        return crop_weights

class REM_ContextEncoderWeighter(nn.Module):
    
    def __init__(self, in_features, sequence_lenght):
        super().__init__()
        self.fc_weights = nn.Sequential(
            nn.Linear(in_features * sequence_lenght, in_features),
            nn.ReLU(),
            nn.Linear(in_features, sequence_lenght),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features):
        features = features.flatten(start_dim = 1)
        return self.fc_weights(features)

############################################################################################
############################################################################################
# ENCODING
############################################################################################
############################################################################################

class REM_CropsEncoder(nn.Module):

    def __init__(self, backbone_name, unification_size):
        super().__init__()
        self.backbone_name = backbone_name
        self.unification_size = unification_size
        self.model = timm.create_model(self.backbone_name, pretrained=True)
        self.fc_proj = nn.Linear(self.model.num_features, self.unification_size)

    def _forward_encoder(self, crop):
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
        encoded_crops = torch.stack([self._forward_encoder(crops[:, i]) for i in range(s)], dim = 1)
        return encoded_crops

############################################################################################
############################################################################################
# CLASSIFICATION
############################################################################################
############################################################################################

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

############################################################################################
############################################################################################
# MAIN MODULE
############################################################################################
############################################################################################

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

        self.crops_weighters = nn.ModuleList([
            REM_CropWeighter(self.unification_size) for _ in range(self.nb_classifiers)
        ])

        self.classifiers = nn.ModuleList([
            REM_LinearClassifier(self.unification_size, self.nb_classes) for _ in range(self.nb_classifiers)
        ])

    def forward_encoders(self, crops):
        features = list()
        for encoder in self.encoders:
            features.append(encoder(crops))
        features = torch.stack(features, dim = 1)
        return features

    def forward(self, crops, mode):
        
        if mode == "train":
            r_b = torch.randint(0, self.nb_encoders, (1,)).item()
            r_w = torch.randint(0, self.nb_classifiers, (1,)).item()
            r_c = torch.randint(0, self.nb_classifiers, (1,)).item()
            backbone = self.encoders[r_b]
            weighter = self.crops_weighters[r_w]
            classifier = self.classifiers[r_c]
            
            encoded = backbone(crops)
            crop_weights = weighter(encoded)
            selected_crops = torch.argmax(crop_weights, dim=1) 
            encoded = torch.einsum("bsf, bs -> bf", encoded, crop_weights)
            output = classifier(encoded)
            return output, crop_weights, selected_crops

        elif mode == "inference":
            encodeds = self.forward_encoders(crops)
            # b n s f
            encoder_weights = list()
            for n in range(self.nb_encoders):
                encoded = encodeds[:, n]
                weights = torch.stack([weighter(encoded) for weighter in self.crops_weighters], dim = 1)
                weights = torch.mean(weights, dim = 1)
                encoder_weights.append(weights)

            encoder_weights = torch.stack(encoder_weights, dim = 1)
            encoder_weights = torch.mean(encoder_weights, dim = 1)

            encodeds = torch.einsum("bnsf, bs -> bf", encodeds, encoder_weights)
            selected_crops = torch.argmax(encoder_weights, dim=1)
        
            classified = torch.stack([classifier(encodeds) for classifier in self.classifiers], dim = 1)
            classified = torch.mean(classified, dim = 1)
            return classified, encoder_weights, selected_crops
    
        else:
            raise Exception("Wrong mode in REM forward !")