import torch
import pydicom
import numpy as np
import cv2
import pandas as pd
import torch.nn as nn
import pickle
import tqdm
import glob
import timm

class DynamicModelLoader:
    def __init__(self, model_name, pretrained=True, hidden_size=256):
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model_name = model_name
        self.hidden_size = hidden_size
        if hidden_size is not None:
            self.model = self.modify_classifier()

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

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
                        'convnext_tiny.in12k_ft_in1k',
                        pretrained=False,
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        features = torch.mean(x, dim = (2, 3))
        return features

class SagittalSliceSelecterModelTorchscript(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.lstm = nn.LSTM(768, 768 // 4, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(768 // 2, 5),
        )

    def forward(self, images):
        _, s, _, _, _ = images.size()
        encoded = torch.stack([self.image_encoder(images[:, i]) for i in range(s)], dim=1)
        lstm_out, _ = self.lstm(encoded)
        out = self.classifier(lstm_out)
        out = out.permute(0, 2, 1)
        return out, torch.softmax(out, dim=2)

def convert_to_ts(f, t):
    with open(f, 'rb') as f:
        model = pickle.load(f)
    
    model = model.eval().cpu()
    
    new_model = SagittalSliceSelecterModelTorchscript()
    new_model.load_state_dict(model.state_dict())
    scripted_model = torch.jit.script(new_model)
    scripted_model.save(t)

if __name__ == "__main__":

    convert_to_ts("../trained_models/v6/model_slice_selection_st2.pkl", "../trained_models/v6/model_slice_selection_st2.ts")