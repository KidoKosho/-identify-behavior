"""
MobileNetV3-small + LSTM cho phân loại hành vi bạo lực.
Input: (B, N, 3, 224, 224) với N = số frame.
Output: (B, 2) logits (violence=1, non-violence=0)
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ViolenceVideoModel(nn.Module):
    def __init__(self, num_frames=16, lstm_hidden_size=128, lstm_num_layers=1, dropout=0.3):
        super().__init__()
        self.num_frames = num_frames

        # Backbone
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # (B, 576, 7, 7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM
        self.lstm = nn.LSTM(
            input_size=576,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (B, N, 3, 224, 224)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        features = self.feature_extractor(x)      # (B*N, 576, 7, 7)
        features = self.avgpool(features)          # (B*N, 576, 1, 1)
        features = features.view(B, N, -1)         # (B, N, 576)

        lstm_out, _ = self.lstm(features)          # (B, N, hidden)
        last_out = lstm_out[:, -1, :]               # (B, hidden)
        logits = self.classifier(last_out)          # (B, 2)
        return logits


def create_violence_model(num_frames=16, device='cpu'):
    model = ViolenceVideoModel(num_frames=num_frames)
    return model.to(device)
