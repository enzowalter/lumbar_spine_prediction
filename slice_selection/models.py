import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
                        'convnext_small.in12k_ft_in1k',
                        pretrained=True,
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        features = torch.mean(x, dim = (2, 3))
        return features

class SliceSelecterModelTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = VIT_Tiny_ImageEncoder()
        self.lstm = nn.LSTM(192, 192 // 4, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(192 // 2, 5),
        )

    def forward(self, images):
        _, s, _, _, _ = images.size()
        encoded = torch.stack([self.image_encoder(images[:, i]) for i in range(s)], dim=1)
        lstm_out, _ = self.lstm(encoded)
        out = self.classifier(lstm_out)
        out = out.permute(0, 2, 1)
        return out.sigmoid()

class VIT_ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
                        "vit_small_patch16_224.augreg_in21k_ft_in1k",
                        pretrained=True,
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x[:, 0]
        return x


class SliceSelecterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = VIT_ImageEncoder()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048), num_layers=2)
        self.classifier = nn.Linear(768, 5)

    def forward(self, images):
        batch_size, seq_len, _, _, _ = images.size()
        encoded = torch.stack([self.image_encoder(images[:, i]) for i in range(seq_len)], dim=1)
        encoded = encoded.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(encoded)
        transformer_out = transformer_out.permute(1, 0, 2)
        out = self.classifier(transformer_out)
        return out.permute(0, 2, 1).sigmoid()

class SliceSelecterModelDistinctClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = VIT_ImageEncoder()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048), num_layers=2)
        self.classifiers = nn.ModuleList([nn.Linear(768, 1) for _ in range(5)])

    def forward(self, images):
        batch_size, seq_len, _, _, _ = images.size()
        encoded = torch.stack([self.image_encoder(images[:, i]) for i in range(seq_len)], dim=1)
        encoded = encoded.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(encoded)
        transformer_out = transformer_out.permute(1, 0, 2)

        outputs = torch.cat([classifier(transformer_out) for classifier in self.classifiers], dim=-1)
        return outputs.permute(0, 2, 1).sigmoid()

class SqueezeNetImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.squeezenet1_0(weights="DEFAULT")

    def forward(self, x):
        x = self.model.features(x)
        features = torch.mean(x, dim = (2, 3))
        features = features.flatten(start_dim=1)
        return features


class AttentionClassifier(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512, batch_first=True), num_layers=2)
        self.fc = nn.Linear(features_size, 1)

    def forward(self, x):
        t = self.transformer(x)
        o = self.fc(t)
        return o

class SliceSelecterModelSqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = SqueezeNetImageEncoder()
        self.attention_classifier = nn.ModuleList([AttentionClassifier(512) for _ in range(5)])

    def forward(self, images):
        batch_size, seq_len, c, h, w = images.size()

        encoded = list()
        for i in range(seq_len):
            if i == 0 or i == seq_len - 1:
                im = images[:, i, ...]
                im = im.expand(batch_size, 3, h, w)
                encoded.append(self.image_encoder(im))
            else:
                im = images[:, i-1:i+2, ...].squeeze(2)
                encoded.append(self.image_encoder(im))

        encoded = torch.stack(encoded, dim = 1)

        outputs = list()
        for attention_classifier in self.attention_classifier:
            outputs.append(attention_classifier(encoded))
        outputs = torch.cat(outputs, dim=2)
        outputs = outputs.permute(0, 2, 1)
        return outputs.sigmoid(), outputs

class SliceSelecterModelTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = VIT_ImageEncoder()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(batch_first=True, d_model=384, nhead=8, dim_feedforward=512), num_layers=2)
        self.classifiers = nn.Linear(384, 5)

    def _encode(self, images):
        batch_size, seq_len, c, h, w = images.size()
        all_images = torch.empty((batch_size, seq_len, 3, h, w), device=images.device).float()
        for i in range(seq_len):
            if i == 0 or i == seq_len - 1:
                all_images[:, i] = images[:, i].repeat(1, 3, 1, 1)
            else:
                triplet = images[:, i-1:i+2].squeeze(2)
                all_images[:, i] = triplet
        encoded_images = self.image_encoder(all_images.view(-1, 3, h, w))
        encoded = encoded_images.view(batch_size, seq_len, -1)
        return encoded

    def forward(self, images):
        encoded = self._encode(images)
        transformer_out = self.transformer_encoder(encoded)
        outputs = self.classifiers(transformer_out)
        outputs = outputs.permute(0, 2, 1)
        return outputs.sigmoid(), outputs

class ConvnextImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
                        "convnext_tiny.fb_in1k",
                        pretrained=True,
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        features = torch.mean(x, dim = (2, 3))
        return features

class VIT_Tiny_ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
                        "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                        pretrained=True,
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x[:, 0]
        return x

class BasicImageEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=True)
    def forward(self, x):
        x = self.model.forward_features(x)
        x = torch.mean(x, dim = (2, 3))
        x = x.flatten(start_dim = 1)
        return x



class REM_SliceSelecterModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = VIT_ImageEncoder()
        self.lstm = nn.LSTM(512, 512 // 4, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 // 2, 5),
        )

    def forward(self, images, mode="inference"):
        _, s, _, _, _ = images.size()
        if mode == "train":

            r_b = torch.randint(0, len(self.encoders), (1,)).item()
            encoder = self.encoders[r_b]

            encoded = torch.stack([encoder(images[:, i]) for i in range(s)], dim=1)
            lstm_out, _ = self.lstm(encoded)
            out = self.classifier(lstm_out)

            out = out.permute(0, 2, 1)
            return out.sigmoid()

        else:
            _, s, _, _, _ = images.size()
            encoded = list()
            for encoder in self.encoders:
                _encoded = torch.stack([encoder(images[:, i]) for i in range(s)], dim=1)
                encoded.append(_encoded)
            encoded = torch.stack(encoded, dim = 1)
            encoded = torch.mean(encoded, dim = 1)
            lstm_out = self.lstm(encoded)[0]
            out = self.classifier(lstm_out)
            out = out.permute(0, 2, 1)
            return out.sigmoid()

class REM_SliceSelecterModel_Scripted(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([SqueezeNetImageEncoder() for _ in range(3)])
        self.lstm = nn.LSTM(512, 512 // 4, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 // 2, 5),
        )
    
    def forward(self, images):
        _, s, _, _, _ = images.size()
        encoded = list()
        for encoder in self.encoders:
            _encoded = torch.stack([encoder(images[:, i]) for i in range(s)], dim=1)
            encoded.append(_encoded)
        encoded = torch.stack(encoded, dim = 1)
        encoded = torch.mean(encoded, dim = 1)
        lstm_out = self.lstm(encoded)[0]
        out = self.classifier(lstm_out)
        out = out.permute(0, 2, 1)
        return out.sigmoid()