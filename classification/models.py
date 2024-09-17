import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

###############################################################################
###############################################################################
# MetaModel from series
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


class ShowShape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print("ShowShape", x.size())
        return x

class DynamicModelLoader:
    def __init__(self, model_name, num_classes=3, pretrained=True, hidden_size=256):
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model_name = model_name
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
        if 'vit_small_patch16_224' in self.model_name:
                self.model.head = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(384, self.hidden_size),
                )
        else:
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

###############################################################################
###############################################################################
# FOLD 1 IMAGE
###############################################################################
###############################################################################

class FoldEncoderAttention(nn.Module):
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
        self.n_classes = n_classes
        
        self.fold_backbones = nn.ModuleList([
            FoldEncoder(features_size, backbone) for backbone in backbones
        ])

        self.fold_classifiers = nn.ModuleList([
            FoldClassifier(features_size, n_classes) for _ in range(n_fold_classifier)
        ])

        self.classifiers_weight = torch.ones((self.n_fold_cla, self.n_fold_enc), dtype=torch.float32)
        self.final_classifier_weight = nn.Parameter(torch.tensor([1. for _ in range(self.n_fold_cla)], dtype=torch.float32))

    def forward_encoders(self, crop):
        encodeds = torch.stack([backbone(crop) for backbone in self.fold_backbones], dim=1)
        return encodeds

    def forward_fold(self, crop, mode):
        if mode == "train":
            r_b = torch.randint(0, self.n_fold_enc, (1,)).item()
            r_c = torch.randint(0, self.n_fold_cla, (1,)).item()
            backbone = self.fold_backbones[r_b]
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
            final_output = list()
            for i in range(self.n_fold_enc):
                enc = self.fold_backbones[i](crop)
                cla = self.fold_classifiers[i](enc)
                final_output.append(cla)
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
        return weighted_output

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
        lstm_out = self.sequence_reducer(lstm_out) # b f
        return lstm_out

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
        return self.forward_fold(crop, mode)

    def forward_fold(self, crop, mode):
        if mode == "train":
            r_b = torch.randint(0, self.n_fold_enc, (1,)).item()
            backbone = self.fold_backbones[r_b]
            classifier = self.fold_classifiers[r_b]
    
            encoded = backbone(crop)
            output = classifier(encoded)
            return output

        if mode == "valid" or mode == "inference":
            final_output = list()
            for i in range(self.n_fold_enc):
                enc = self.fold_backbones[i](crop)
                cla = self.fold_classifiers[i](enc)
                final_output.append(cla)
            final_output = torch.mean(torch.stack(final_output, dim=1), dim=1)
            return final_output


###############################################################################
###############################################################################
# Models Dual Output
###############################################################################
###############################################################################

class MaskGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 7 7
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.021),
            # 14 14
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.021),
            # 28 28
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.021),
            # 56 56
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 5, 1, 1, 0)
        )
    def forward(self, x):
        x = self.main(x)
        x = F.interpolate(x, size=(96, 96), mode="bilinear")
        x = self.outconv(x)
        return x.sigmoid()


class MobileNetDualOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights="DEFAULT")
        self.mask_generator = MaskGenerator()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 15)
        )

    def forward_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x  

    def forward(self, x):
        features = self.forward_features(x)
        mask = self.mask_generator(features)
        labels = self.classifier(features)
        return labels, mask

###############################################################################
###############################################################################
# Basic Classification Models
###############################################################################
###############################################################################

class MetaModelClassifier(nn.Module):
    def __init__(self, feature_size, seq_lenght, n_classes):
        super().__init__()
        self.feature_size = feature_size
        self.seq_lenght = seq_lenght
        self.n_classes = n_classes

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * self.seq_lenght, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, self.n_classes),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), self.feature_size * self.seq_lenght)
        x = self.classifier(x)
        return x

class SimpleClassifier(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.model = DynamicModelLoader(model_name=encoder_name, hidden_size=256).model
        self.classifier = nn.Linear(256, 3)
    
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


class Resnet3dClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.video.r3d_18(weights="DEFAULT")
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward_features(self, x):
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.flatten(1)
        return x

    def forward(self, x):
        b, s, h, w = x.size()
        x = x.unsqueeze(1)
        x = x.expand(b, 3, s, h, w)
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

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
    
    def forward_features(self, x):
        x = self.backbone.features(x)
        x = self.pooler(x)
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

class SEResNetClassification(nn.Module):
    def __init__(self, num_classes=3):
        super(SEResNetClassification, self).__init__()
        self.backbone = timm.create_model('seresnet50', pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
