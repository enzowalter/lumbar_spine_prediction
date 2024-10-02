import torch
import torch.nn as nn
import timm

class REM_Series_Classifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.021),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, n_classes)
        )

    def forward(self, encodeds):
        out = self.classifier(encodeds)
        return out

class REM_Series_Encoder(nn.Module):
    def __init__(self, backbone_name, out_size):
        super().__init__()
        self.name = backbone_name
        self.out_size = out_size
        self.model = timm.create_model(model_name=backbone_name, pretrained=True)
        self.projection = nn.Linear(self.model.num_features, out_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.out_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

    def _forward(self, crops):
        b, h, w = crops.size()
        crops = crops.unsqueeze(1).expand(b, 3, h, w)
        encoded = self.model.forward_features(crops)
        if len(encoded.shape) == 3: # transformer b s f
            encoded = encoded[:, 0]
        else:
            encoded = torch.mean(encoded, dim = (2, 3))
            encoded = encoded.flatten(start_dim=1)
            
        encoded = self.projection(encoded)
        return encoded

    def forward(self, crops):
        b, c, h, w = crops.size()
        encodeds = torch.stack([self._forward(crops[:, i]) for i in range(c)], dim = 1)
        trans_out = self.transformer_encoder(encodeds)
        trans_out, _ = torch.max(trans_out, dim = 1)
        return trans_out

class REM(nn.Module):
    def __init__(self, backbones, nb_classifiers, unification_features_size, n_classes):
        super().__init__()
        self.nb_encoders = len(backbones)
        self.nb_classifiers = nb_classifiers
        self.n_classes = n_classes
        self.unification_features_size = unification_features_size
        
        self.encoders = nn.ModuleList([
            REM_Series_Encoder(backbone, self.unification_features_size) for backbone in backbones
        ])
        self.classifiers = nn.ModuleList([
            REM_Series_Classifier(self.unification_features_size, n_classes) for _ in range(nb_classifiers)
        ])
        self.weights_encoders_per_classifier = nn.Parameter(torch.ones(nb_classifiers, self.nb_encoders))

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
    
        if mode == "train_gate":
            final_output = list()
            
            with torch.no_grad():
                _encodeds = self.forward_encoders(crop)
    
            for idx, classifier in enumerate(self.classifiers):
                classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
                classifier_weights = torch.softmax(self.weights_encoders_per_classifier[idx], dim=0)
                classifier_output = torch.einsum("bsf,s->bf", classified_, classifier_weights)
                final_output.append(classifier_output)
            final_output = torch.stack(final_output, dim=1)
            final_output = torch.mean(final_output, dim=1)
            return final_output

        if mode == "inference":
            final_output = list()
            _encodeds = self.forward_encoders(crop)
            for idx, classifier in enumerate(self.classifiers):
                classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
                classifier_weights = torch.softmax(self.weights_encoders_per_classifier[idx], dim=0)
                classifier_output = torch.einsum("bsf,s->bf", classified_, classifier_weights)
                final_output.append(classifier_output)
            final_output = torch.stack(final_output, dim=1)
            final_output = torch.mean(final_output, dim=1)
            return final_output
