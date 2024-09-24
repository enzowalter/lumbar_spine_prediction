import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
                        'vit_small_patch16_224.augreg_in21k_ft_in1k',
                        pretrained=True,
        )

        # tiny = 0.9098076372398568 f1
        # small = 0.9265487995731697 f1
        # base = 0.921396780326511

    def forward(self, x):
        x = self.model.forward_features(x)[:, 0]
        return x

class AxialSliceClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.ReLU(),
            nn.Linear(192, 6)
        )

    def forward(self, images):
        features = self.image_encoder(images)
        out = self.classifier(features)
        return out

