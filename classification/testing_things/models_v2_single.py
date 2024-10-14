import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GCNConv

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

    def __init__(self, backbone_name):
        super().__init__()
        self.backbone_name = backbone_name
        # encoding
        self.model = timm.create_model(self.backbone_name, pretrained=True)
        self.num_features = self.model.num_features
        # crop selection
        self.policy = REM_Policy(self.num_features)

    def _forward(self, crop):
        encoded = self.model.forward_features(crop)
        if len(encoded.shape) == 3: # transformer b s f
            encoded = encoded[:, 0]
        else:
            encoded = torch.mean(encoded, dim = (2, 3))
            encoded = encoded.flatten(start_dim=1)
        return encoded

    def forward(self, crops):
        # return self._forward(crops)
        b, s, c, h, w = crops.size()
        encoded_crops = torch.stack([self._forward(crops[:, i]) for i in range(s)], dim = 1)
        crops_probs = self.policy(encoded_crops)
        selected_crops = torch.argmax(crops_probs, dim=1) 
        features = torch.einsum("bsf, bs -> bf", encoded_crops, crops_probs)
        return features, crops_probs, selected_crops

class REM_CNNClassifier(nn.Module):
    def __init__(self, features_size, nb_classes):
        super(REM_CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding = 1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding = 1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * features_size // 4, 128)
        self.fc2 = nn.Linear(128, nb_classes)
    
    def forward(self, features):
        features = features.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(features)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class REM_Single(nn.Module):
    def __init__(self, n_classes, backbone):
        super().__init__()
        self.nb_classes = n_classes
        self.encoder = REM_CropsEncoder(backbone)
        self.classifiers = REM_CNNClassifier(self.encoder.num_features, self.nb_classes)

    def forward(self, crops):
        features, crops_probs, selected_crops = self.encoder(crops)
        output = self.classifiers(features)
        # return output
        return output, crops_probs, selected_crops
