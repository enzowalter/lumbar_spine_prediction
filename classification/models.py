import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ExpertEncoder(nn.Module):
    def __init__(self, encoder_name, out_features, sequence_lenght, backbone_features=1280):
        super().__init__()
        self.sequence_lenght = sequence_lenght
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
        out = torch.stack([self._forward_image(images[:, i]) for i in range(self.sequence_lenght)], dim=1)
        return out

class ExpertClassifier(nn.Module):
    def __init__(self, in_features, sequence_lenght, num_classes):
        super().__init__()
        self.sequence_lenght = sequence_lenght
        self.in_features = in_features

        self.gating_encoder = nn.Sequential(
            nn.Linear(self.in_features, 1),
            nn.Sigmoid()
        )

        in_size = in_features * sequence_lenght
        self.classifier = nn.Sequential(
            nn.Linear(in_size, in_size // 4),
            nn.ReLU(),
            nn.Linear(in_size // 4, num_classes)
        )

    def forward(self, encoded):
        # shape = (batch_size, sequence_lenght, self.encoder_features)
        gates_encoder = torch.sigmoid(self.gating_encoder(encoded))
        encoder_output = torch.sum(gates_encoder * encoded, dim=1)
        # encoder_output.shape = (batch_size, self.sequence_lenght, self.encoder_features)

        encoder_output = encoder_output.reshape(encoded.shape[0], self.sequence_lenght * self.in_features)
        return self.classifier(encoder_output)

class ExperCeption(nn.Module):
    def __init__(self, 
                num_expert_encoder=3,
                num_expert_classifier=3,
                encoder_features=64,
                num_classes=3,
                sequence_lenght=5,
                encoder="efficientnet"
                ):
        super().__init__()
        self.num_expert_encoder = num_expert_encoder    
        self.num_expert_classifier = num_expert_classifier    
        self.encoder_features = encoder_features    
        self.num_classes = num_classes    
        self.sequence_lenght = sequence_lenght
        self.encoder = encoder

        self.experts_encoder = nn.ModuleList([ExpertEncoder(encoder_name=encoder, out_features=self.encoder_features, sequence_lenght=self.sequence_lenght) for _ in range(self.num_expert_encoder)])
        self.experts_classifier = nn.ModuleList([ExpertClassifier(in_features=self.encoder_features, sequence_lenght=self.sequence_lenght, num_classes=self.num_classes) for _ in range(self.num_expert_classifier)])

        self.gating_classifier = nn.Sequential(
            nn.Linear(self.num_classes, 1),
            nn.Sigmoid()
        )

    def diversity_loss(self, num_experts, expert_outputs):
        if num_experts > 1:
            loss = 0.0
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    cos_sim = F.cosine_similarity(expert_outputs[:, i], expert_outputs[:, j], dim=-1)
                    loss += cos_sim.mean()
            return loss / (num_experts * (num_experts - 1) / 2)
        return 0

    def forward(self, images):
        expert_encoder_output = torch.stack([self.experts_encoder[i](images) for i in range(self.num_expert_encoder)], dim=1)
        loss_diversity_encoder = self.diversity_loss(self.num_expert_encoder, expert_encoder_output)
        # expert_encoder_output.shape = (batch_size, self.num_expert_encoder, self.sequence_lenght, self.encoder_features)

        expert_classifier_output = torch.stack([self.experts_classifier[i](expert_encoder_output) for i in range(self.num_expert_classifier)], dim = 1)
        loss_diversity_classifier = self.diversity_loss(self.num_expert_classifier, expert_classifier_output)
        # expert_classifier_output.shape = (batch_size, self.num_expert_classifier, self.sequence_lenght * self.num_classes)

        gates_classifier = torch.sigmoid(self.gating_classifier(expert_classifier_output))
        classifier_output = torch.sum(gates_classifier * expert_classifier_output, dim=1)
        # classifier_output.shape = (batch_size, self.num_classes)

        return classifier_output, loss_diversity_encoder + loss_diversity_classifier