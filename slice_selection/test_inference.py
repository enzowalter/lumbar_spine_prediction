import torch
import torch.nn as nn
import timm
import time
import torch.nn.functional as F
import torch.nn.utils.prune as prune

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

# class SliceSelecterModelTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.image_encoder = VIT_ImageEncoder()
#         self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(batch_first=True, d_model=384, nhead=8, dim_feedforward=512), num_layers=2)
#         self.classifiers = nn.Linear(384, 5)

#     def _encode(self, images):
#         batch_size, seq_len, c, h, w = images.size()
#         all_images = torch.empty((batch_size, seq_len, 3, h, w), device=images.device).float()
#         for i in range(seq_len):
#             if i == 0 or i == seq_len - 1:
#                 all_images[:, i] = images[:, i].repeat(1, 3, 1, 1)
#             else:
#                 triplet = images[:, i-1:i+2].squeeze(2)
#                 all_images[:, i] = triplet
#         encoded_images = self.image_encoder(all_images.view(-1, 3, h, w))
#         encoded = encoded_images.view(batch_size, seq_len, -1)
#         return encoded

#     def forward(self, images):
#         encoded = self._encode(images)
#         transformer_out = self.transformer_encoder(encoded)
#         outputs = self.classifiers(transformer_out)
#         outputs = outputs.permute(0, 2, 1)
#         return outputs.sigmoid(), outputs

# import time

# model = SliceSelecterModelTransformer()
# model = model.cuda()
# model = model.eval()
# inputs = torch.randn((1, 30, 1, 224, 224)).cuda()

# with torch.no_grad():
#     start = time.time()
#     for _ in range(50):
#         _ = model(inputs)
#     print("Time 50 inference:", time.time() - start)
#     start = time.time()
#     for _ in range(50):
#         _ = model(inputs)

#     print("Time 50 inference:", time.time() - start)
#     start = time.time()
#     for _ in range(50):
#         _ = model(inputs)

#     print("Time 50 inference:", time.time() - start)


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

        self.weights = nn.Parameter(torch.ones(self.nb_classifiers, self.nb_encoders))

    def forward_encoders(self, crops):
        batch_size = crops.size(0)
        seq_length = crops.size(1)
        features_shape = (batch_size, self.nb_encoders, self.unification_size)
        weights_shape = (batch_size, self.nb_encoders, seq_length)
        selected_shape = (batch_size, self.nb_encoders)

        features = torch.empty(features_shape, device=crops.device, dtype=crops.dtype)
        weights = torch.empty(weights_shape, device=crops.device, dtype=crops.dtype)
        selected = torch.empty(selected_shape, device=crops.device, dtype=crops.dtype)

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
    
        if mode == "inference":
            _encodeds, weights, selected = self.forward_encoders(crops)
            classified_outputs = torch.stack([classifier(_encodeds) for classifier in self.classifiers], dim=1)
            final_output = classified_outputs.mean(dim=(1, 2))
            return final_output, weights, selected

def prune_model(model, amount=0.2):

    # Prune the weights of the fc_proj in REM_CropsEncoder for each encoder
    for i, encoder in enumerate(model.encoders):
        prune.random_unstructured(encoder.fc_proj, name='weight', amount=amount)  # fc_proj layer
        print(f"Pruned {amount * 100:.1f}% of the weights in fc_proj for encoder {i}.")
        
        # Prune the weights of the fc_weights in the REM_Policy
        prune.random_unstructured(encoder.policy.fc_weights[1], name='weight', amount=amount)  # Second linear layer
        print(f"Pruned {amount * 100:.1f}% of the weights in fc_weights[1].")

        # Prune layers in the backbone model
        for layer in encoder.model.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):  # Prune only Conv and Linear layers
                prune.random_unstructured(layer, name='weight', amount=amount)
                print(f"Pruned {amount * 100:.1f}% of the weights in backbone {encoder.backbone_name} for layer {layer}.")

    # Prune the classifier layers in REM_LinearClassifier
    for i, classifier in enumerate(model.classifiers):
        prune.random_unstructured(classifier.classifier[1], name='weight', amount=amount)  # First linear layer in classifier
        print(f"Pruned {amount * 100:.1f}% of the weights in classifier {i}.")

backbones = ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k']
# backbones = ['convnext_tiny.in12k', 'focalnet_tiny_lrf.ms_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'cspresnet50.ra_in1k']

model = REM(
    n_classes=3,
    n_classifiers=3,
    unification_size=512,
    backbones=backbones,
)

model = model.cuda()
model = model.eval()
model = model.half()
inputs = torch.randn((1, 5, 3, 128, 128)).cuda()
inputs = inputs.half()

with torch.no_grad():
    start = time.time()
    for _ in range(50):
        _ = model(inputs, mode = "inference")
    print("Time 50 inference:", time.time() - start)

#     print("Time 50 inference:", time.time() - start)
#     start = time.time()
#     for _ in range(50):
#         _ = model(inputs, mode = "inference")

#     print("Time 50 inference:", time.time() - start)