import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

###############################################################################
###############################################################################
# MetaModel
###############################################################################
###############################################################################

class MetaModelAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.query = nn.Linear(features, features)
        self.key = nn.Linear(features, features)
        self.value = nn.Linear(features, features)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoded):
        query = self.query(encoded)
        key = self.key(encoded)
        value = self.value(encoded)

        attention_scores = torch.einsum('bsf,bsf->bs', query, key) / (encoded.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        weighted_sum = torch.einsum('bsf,bs->bf', value, attention_weights)

        return weighted_sum

class CropClassifierMetaModel(nn.Module):
    def __init__(self, encoder_models):
        super().__init__()

        self.encoder_models = nn.ModuleList(encoder_models)
        self.attention_encoder = nn.ModuleList(MetaModelAttention(256) for _ in range(len(encoder_models)))
        self.weights_encoders = nn.Parameter(torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32))
        self.classifier = nn.Sequential(
            nn.Dropout(0.21),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    
    def _forward_encoder(self, crops, i):
        out = self.encoder_models[i].forward_encoders(crops[:, i])
        out = self.attention_encoder[i](out)
        return out

    def forward(self, crops):
        b, s, c, h, w = crops.size()
        output_encoder = torch.zeros((b, s, 256)).to(crops.device)
        for i in range(5):
            output_encoder[:, i] = self._forward_encoder(crops, i)
        
        normalized_weights = torch.softmax(self.weights_encoders, dim=0)
        weighted_encoders = torch.einsum("bsf,s->bf", output_encoder, normalized_weights)
        output = self.classifier(weighted_encoders)
        return output

###############################################################################
###############################################################################
# IDK
###############################################################################
###############################################################################

class IDKEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b0(weights="DEFAULT")
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, 256)
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.pooler(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class IDKAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.query = nn.Linear(features, features)
        self.key = nn.Linear(features, features)
        self.value = nn.Linear(features, features)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoded):
        query = self.query(encoded)
        key = self.key(encoded)
        value = self.value(encoded)

        attention_scores = torch.einsum('bsf,bsf->bs', query, key)
        #attention_scores = torch.einsum('bsf,bsf->bs', query, key) / (encoded.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        weighted_sum = torch.einsum('bsf,bs->bf', value, attention_weights)

        return weighted_sum

class IDKModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([IDKEncoder() for _ in range(5)])
        self.attn = IDKAttention(256)
        self.fc = nn.Sequential(
            nn.Dropout(0.021),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, crops, encoder_index):
        b, s, c, h, w = crops.size()

        if encoder_index < 5:
            encoded = self.encoders[encoder_index](crops[:, encoder_index])
        else:
            encoded = torch.zeros(b, s, 256).to(crops.device)
            for i in range(5):
                encoded[:, i] = self.encoders[i](crops[:, i])
            encoded = self.attn(encoded)

        output = self.fc(encoded)
        return output

###############################################################################
###############################################################################
# EXPERCEPTION => BETTER F1 but LOWER LogLoss
###############################################################################
###############################################################################

class ExpertEncoder(nn.Module):
    def __init__(self, encoder_name, out_features):
        super().__init__()
        self.out_features = out_features

        if encoder_name == "efficientnet":
            self.encoder = torchvision.models.efficientnet_b0(weights="DEFAULT")
            self.classifier = nn.Linear(1280, out_features)
        elif encoder_name == "squeezenet":
            self.encoder = torchvision.models.squeezenet1_0(weights="DEFAULT")
            self.classifier = nn.Linear(512, out_features)
        else:
            raise NotImplementedError("Unknown encoder")

    def _forward_image(self, image):
        features = self.encoder.features(image)
        features = features.mean(dim=(2, 3))
        out = self.classifier(features)
        return out

    def forward(self, images):
        b, s, c, h, w = images.size()
        out = torch.zeros(b, s, self.out_features).to(images.device)
        for i in range(s):
            out[:, i] = self._forward_image(images[:, i])
        return out

class GatingEncoder(nn.Module):
    def __init__(self, ich, och, features):
        super().__init__()
        self.conv = nn.Conv1d(ich, och, 1, 1, 0)
        self.act = nn.ReLU()
        self.fc = nn.Linear(features, 1)

    def _forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def forward(self, encoded):
        b, n, s, f = encoded.size()
        weights = torch.cat([self._forward(encoded[:, i]) for i in range(n)], dim = 1)
        weights = weights.sigmoid()
        encoded = torch.einsum("bnsf,bn->bsf", encoded, weights)
        return encoded

class ExperCeption(nn.Module):
    def __init__(self, 
                num_expert_encoder=5,
                encoder_features=128,
                num_classes=3,
                seq_lenght=5,
                encoder="efficientnet"
                ):
        super().__init__()
        self.num_expert_encoder = num_expert_encoder    
        self.encoder_features = encoder_features    
        self.num_classes = num_classes    
        self.encoder = encoder
        self.seq_lenght = seq_lenght

        self.experts_encoder = nn.ModuleList([
                            ExpertEncoder(
                                encoder_name=encoder, 
                                out_features=self.encoder_features
                            ) 
                            for _ in range(self.num_expert_encoder)
                        ])

        self.gating_encoder = GatingEncoder(seq_lenght, 1, self.encoder_features)

        self.lstm = nn.LSTM(
            self.encoder_features, 
            self.encoder_features // 2, 
            bidirectional=True,
            batch_first=True)
    
        self.fc = nn.Linear(self.encoder_features, 3)

    def forward(self, images):
        b, s, c, h, w = images.size()
        device = images.device
        
        expert_encoder_output = torch.zeros(b, self.num_expert_encoder, s, self.encoder_features).to(device)
        for i, expert in enumerate(self.experts_encoder):
            expert_encoder_output[:, i] = expert(images)
        output_encoder = self.gating_encoder(expert_encoder_output)
        lstm_out, _ = self.lstm(output_encoder)
        classifier_output = self.fc(lstm_out)
        return classifier_output

class EfficientNetClassifierSeries(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b0(weights="DEFAULT")
        self.lstm = nn.LSTM(1280, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, 3)

    def _encoder_forward(self, x):
        x = self.backbone.features(x)
        x = torch.mean(x, dim = (2, 3))
        return x

    def forward(self, crops):
        b, s, c, h, w = crops.size()
        encoded = torch.stack([self._encoder_forward(crops[:, i]) for i in range(s)], dim = 1)
        lstm_out, _ = self.lstm(encoded)
        classified = self.classifier(lstm_out)
        return classified

class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b0(weights="DEFAULT")
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.021),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, crop):
        encoded = self.backbone.features(crop)
        encoded = self.pooler(encoded)
        encoded = encoded.view(encoded.shape[0], -1)
        classified = self.classifier(encoded)
        return classified
    
class DynamicModelLoader:
    def __init__(self, model_name, num_classes=3, pretrained=True, hidden_size=256):
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.feature_size = self.get_feature_size()
        self.model = self.modify_classifier()

    def get_feature_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.model.forward_features(dummy_input)
            return features.shape[1]
    
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
        else:
            raise NotImplementedError("Unknown classifier structure")
        return self.model

###############################################################################
###############################################################################
# FOLD 1 IMAGE
###############################################################################
###############################################################################

class FoldEncoder(nn.Module):
    def __init__(self, features_size, backbone_name):
        super().__init__()
        self.name = backbone_name
        self.model = DynamicModelLoader(model_name=backbone_name, hidden_size=features_size).model

    def forward(self, x):
        return self.model(x)

class FoldClassifier(nn.Module):
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

class FoldModelClassifier(nn.Module):
    def __init__(self, backbones, n_fold_classifier, features_size, n_classes):
        super().__init__()
        self.n_fold_enc = len(backbones)
        self.n_fold_cla = n_fold_classifier
        self.features_size = features_size
        
        self.fold_backbones = nn.ModuleList([
            FoldEncoder(features_size, backbone) for backbone in backbones
        ])

        self.fold_classifiers = nn.ModuleList([
            FoldClassifier(features_size, n_classes) for _ in range(n_fold_classifier)
        ])

    def forward_encoders(self, crop):
        encodeds = torch.stack([backbone(crop) for backbone in self.fold_backbones], dim=1)
        return encodeds

    def forward(self, crop, mode):
        if mode == "train":
            r_b = torch.randint(0, self.n_fold_enc, (1,)).item()
            backbone = self.fold_backbones[r_b]

            r_c = torch.randint(0, self.n_fold_cla, (1,)).item()
            classifier = self.fold_classifiers[r_c]
    
            encoded = backbone(crop)
            output = classifier(encoded)
            return output

        if mode == "valid" or mode == "inference":
            final_output = list()
            _encodeds = self.forward_encoders(crop)
            for classifier in self.fold_classifiers:
                classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.n_fold_enc)], dim=1)
                classifier_output = torch.mean(classified_, dim=1)
                final_output.append(classifier_output)

            final_output = torch.stack(final_output, dim=1)
            final_output = torch.mean(final_output, dim=1)
            return final_output

###############################################################################
###############################################################################
# FOLD SERIES
###############################################################################
###############################################################################

class ReduceWithAttention(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.attention = nn.Linear(features_size, 1, bias=False)
    
    def forward(self, lstm_out):
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        weighted_output = torch.sum(lstm_out * attn_weights, dim=1)
        return weighted_output, attn_weights

class FoldClassifierSeries(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, n_classes)
        )

    def forward(self, x):
        out = self.fc(x) # n_class
        return out

class FoldEncoderSeries(nn.Module):
    def __init__(self, seq_lenght, features_size, backbone_name):
        super().__init__()
        self.seq_lenght = seq_lenght
        self.name = backbone_name
        self.model = DynamicModelLoader(model_name=backbone_name, hidden_size=features_size).model
        self.lstm = nn.LSTM(features_size, features_size // 2, batch_first=True, bidirectional=True, num_layers=2)
        self.sequence_reducer = ReduceWithAttention(features_size)

    def forward(self, x):
        encodeds = []
        for n in range(self.seq_lenght):
            encodeds.append(self.model(x[:, n]))
        encodeds = torch.stack(encodeds, dim = 1) # => b, s, f
        lstm_out, _ = self.lstm(encodeds) # b s f
        lstm_out, attention_weight = self.sequence_reducer(lstm_out) # b f
        return lstm_out, attention_weight

class FoldModelClassifierFromSeries(nn.Module):
    def __init__(self, n_fold_classifier, backbones, seq_lenght, features_size, n_classes):
        super().__init__()
        self.n_fold_enc = len(backbones)
        self.n_fold_cla = n_fold_classifier
        self.features_size = features_size

        self.fold_backbones = nn.ModuleList([
            FoldEncoderSeries(seq_lenght, features_size, backbone) for backbone in backbones
        ])

        self.fold_classifiers = nn.ModuleList([
            FoldClassifierSeries(features_size, n_classes) for _ in range(self.n_fold_cla)
        ])

    def forward(self, crop, mode):
        if mode == "train":
            r_b = torch.randint(0, self.n_fold_enc, (1,)).item()
            backbone = self.fold_backbones[r_b]

            r_c = torch.randint(0, self.n_fold_cla, (1,)).item()
            classifier = self.fold_classifiers[r_c]
    
            encoded, attention_weight = backbone(crop)
            output = classifier(encoded)
            return output, attention_weight

        if mode == "valid" or mode == "inference":
            final_output = list()
            _encodeds = torch.stack([backbone(crop)[0] for backbone in self.fold_backbones], dim=1)
            for classifier in self.fold_classifiers:
                classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.n_fold_enc)], dim=1)
                classifier_output = torch.mean(classified_, dim=1)
                final_output.append(classifier_output)

            final_output = torch.stack(final_output, dim=1)
            final_output = torch.mean(final_output, dim=1)
            return final_output

###############################################################################
###############################################################################
# EfficientNet Classification
###############################################################################
###############################################################################

class EfficientNetClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b0(weights="DEFAULT")
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.pooler(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
import torchvision
import timm

class SEResNetClassification(nn.Module):
    def __init__(self, num_classes=3):
        super(SEResNetClassification, self).__init__()
        # Load pretrained SE-ResNet50 model from timm
        self.backbone = timm.create_model('seresnet50', pretrained=True)
        
        # Replace the final fully connected layer for your classification task
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)