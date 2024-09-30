import torch
import torch.nn as nn
import timm

class REM_AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(REM_AttentionModule, self).__init__()
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tensor_sequence):
        batch_size, seq_len, feature_dim = tensor_sequence.size()
        queries = self.query_layer(tensor_sequence)
        keys = self.key_layer(tensor_sequence)
        values = self.value_layer(tensor_sequence)
        attention_scores = torch.einsum('bqd,bkd->bqk', queries, keys)
        attention_weights = self.softmax(attention_scores)
        attention = attention_weights.sum(dim=-1)
        output = torch.einsum('bsf, bs -> bf', values, attention)
        return output

class REM_Encoder(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.name = backbone_name
        self.model = timm.create_model(model_name=backbone_name, pretrained=True)

    def forward(self, crops):
        encoded = self.model.forward_features(crops)
        if len(encoded.shape) == 3: # transformer b s f
            encoded = encoded[:, 0]
        else:
            encoded = torch.mean(encoded, dim = (2, 3))
            encoded = encoded.flatten(start_dim=1)
        return encoded

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
    def __init__(self, backbone_name):
        super().__init__()
        self.name = backbone_name
        self.model = timm.create_model(model_name=backbone_name, pretrained=True)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
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
        return encoded

    def forward(self, crops):
        b, c, h, w = crops.size()
        encodeds = torch.stack([self._forward(crops[:, i]) for i in range(c)], dim = 1)
        trans_out = self.transformer_encoder(encodeds)
        trans_out, _ = torch.max(trans_out, dim = 1)
        return trans_out

class REM(nn.Module):
    def __init__(self, backbones, n_fold_classifier, n_classes):
        super().__init__()
        self.nb_encoders = len(backbones)
        self.nb_classifiers = n_fold_classifier
        self.n_classes = n_classes
        
        self.encoders = nn.ModuleList([
            REM_Series_Encoder(backbone) for backbone in backbones
        ])
        self.classifiers = nn.ModuleList([
            REM_Series_Classifier(1024, n_classes) for _ in range(n_fold_classifier)
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

class REM_Script(nn.Module):
    def __init__(self, backbones, n_fold_classifier, n_classes):
        super().__init__()
        self.nb_encoders = len(backbones)
        self.nb_classifiers = n_fold_classifier
        self.n_classes = n_classes
        
        self.encoders = nn.ModuleList([
            REM_Encoder(backbone) for backbone in backbones
        ])
        self.classifiers = nn.ModuleList([
            REM_Classifier(1024, n_classes) for _ in range(n_fold_classifier)
        ])
        self.weights_encoders = nn.Parameter(torch.ones(len(self.encoders)))

    def forward_encoders(self, crop):
        encodeds = torch.stack([encoder(crop) for encoder in self.encoders], dim=1)
        return encodeds

    def forward(self, crop):
        final_output = list()
        _encodeds = self.forward_encoders(crop)
        for classifier in self.classifiers:
            classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
            classifier_output = torch.einsum("bsf,s->bf", classified_, torch.softmax(self.weights_encoders, dim=0))
            final_output.append(classifier_output)
        final_output = torch.stack(final_output, dim=1)
        final_output = torch.mean(final_output, dim=1)
        return final_output


class REM_5x(nn.Module):
    def __init__(self, backbones, n_fold_classifier, n_classes):
        super().__init__()
        self.nb_encoders = len(backbones)
        self.nb_classifiers = n_fold_classifier
        self.n_classes = n_classes
        
        self.encoders = nn.ModuleList([
            REM_5x_Encoder(backbone) for backbone in backbones
        ])
        self.classifiers = nn.ModuleList([
            REM_Classifier(1024, n_classes) for _ in range(n_fold_classifier)
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

class REM_5x_Script(nn.Module):
    def __init__(self, backbones, n_fold_classifier, n_classes):
        super().__init__()
        self.nb_encoders = len(backbones)
        self.nb_classifiers = n_fold_classifier
        self.n_classes = n_classes
        
        self.encoders = nn.ModuleList([
            REM_5x_Encoder(backbone) for backbone in backbones
        ])
        self.classifiers = nn.ModuleList([
            REM_Classifier(1024, n_classes) for _ in range(n_fold_classifier)
        ])
        self.weights_encoders = nn.Parameter(torch.ones(len(self.encoders)))

    def forward_encoders(self, crop):
        encodeds = torch.stack([encoder(crop) for encoder in self.encoders], dim=1)
        return encodeds

    def forward(self, crop):
        final_output = list()
        _encodeds = self.forward_encoders(crop)
        for classifier in self.classifiers:
            classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
            classifier_output = torch.einsum("bsf,s->bf", classified_, torch.softmax(self.weights_encoders, dim=0))
            final_output.append(classifier_output)
        final_output = torch.stack(final_output, dim=1)
        final_output = torch.mean(final_output, dim=1)
        return final_output
