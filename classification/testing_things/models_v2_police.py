import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class REM_Policy_Light(nn.Module):
    def __init__(self, feature_size, n_heads=4):
        super().__init__()
        
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=n_heads,
            batch_first=True,
        )
        self.fc_weights = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, 1)
        )

    def forward(self, encodeds):
        attention_output, _ = self.multihead_attention(encodeds, encodeds, encodeds)
        attention_scores = self.fc_weights(attention_output)
        attention_scores = attention_scores.squeeze(-1)
        crop_weights = torch.softmax(attention_scores, dim=1)
        return crop_weights

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
        return encoded_crops

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

    def __init__(self, n_classes, backbones, unification_size):
        super().__init__()

        self.nb_classes = n_classes
        self.nb_encoders = len(backbones)
        self.backbones = backbones
        self.unification_size = unification_size

        self.encoders = nn.ModuleList([
            REM_CropsEncoder(backbone, self.unification_size) for backbone in self.backbones
        ])

        self.policy = REM_Policy_Light(self.unification_size)

        self.classifiers = nn.ModuleList([
            REM_LinearClassifier(self.unification_size, self.nb_classes) for _ in range(len(self.backbones))
        ])

        self.context_weighting_net = nn.Sequential(
            nn.Linear(self.unification_size * self.nb_encoders, self.unification_size),
            nn.ReLU(),
            nn.Linear(self.unification_size, self.nb_encoders),
            nn.Softmax(dim=1)
        )

    def forward_encoders(self, crops):
        features = torch.stack([encoder(crops) for encoder in self.encoders], dim = 1)
        return features
    
    def forward(self, crops, mode):
        
        if mode == "train":
            random_index = torch.randint(0, self.nb_encoders, (1,)).item()
            backbone = self.encoders[random_index]
            classifier = self.classifiers[random_index]

            encoded_crops = backbone(crops)
            weight_crops = self.policy(encoded_crops)
            selected_crops = torch.argmax(weight_crops, dim=1) 

            weighted_crops = torch.einsum("bsf, bs -> bf", encoded_crops, weight_crops)
            outputs = classifier(weighted_crops)
            return outputs, weight_crops, selected_crops
    
        if mode == "train_gate":
            with torch.no_grad():
                encodeds = self.forward_encoders(crops)
                weight_crops = torch.stack([self.policy(encodeds[:, i]) for i in range(self.nb_encoders)], dim = 1)
                weight_crops = torch.mean(weight_crops, dim = 1)
                selected_crops = torch.argmax(weight_crops, dim=1)
                weighted_crops = torch.einsum("bnsf, bs -> bnf", encodeds, weight_crops)
                outputs = torch.stack([self.classifiers[i](weighted_crops[:, i]) for i in range(self.nb_encoders)], dim = 1)

            aggregated_features = weighted_crops.view(weighted_crops.size(0), -1)
            context_weights = self.context_weighting_net(aggregated_features)
            outputs = torch.einsum("bnf, bn -> bf", outputs, context_weights)
            return outputs, weight_crops, selected_crops

        if mode == "inference":
            encodeds = self.forward_encoders(crops)
            weight_crops = torch.stack([self.policy(encodeds[:, i]) for i in range(self.nb_encoders)], dim = 1)
            weight_crops = torch.mean(weight_crops, dim = 1)
            selected_crops = torch.argmax(weight_crops, dim=1)
            weighted_crops = torch.einsum("bnsf, bs -> bnf", encodeds, weight_crops)

            outputs = torch.stack([self.classifiers[i](weighted_crops[:, i]) for i in range(self.nb_encoders)], dim = 1)
            aggregated_features = weighted_crops.view(weighted_crops.size(0), -1)
            context_weights = self.context_weighting_net(aggregated_features)
            outputs = torch.einsum("bnf, bn -> bf", outputs, context_weights)
            return outputs, weight_crops, selected_crops
