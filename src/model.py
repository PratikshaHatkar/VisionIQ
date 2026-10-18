import torch
import torch.nn as nn
import torchvision.models.video as vmodels
from torchvision.models.video import R2Plus1D_18_Weights, MC3_18_Weights
import torchvision.models as models

from src.config import NUM_CLASSES

class ActionRecognitionModel(nn.Module):
    """
    Spatiotemporal Video Action Recognition Model based on R(2+1)D-18
    pretrained on Kinetics-400 with fine-tuning head.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_early_layers=False):
        super(ActionRecognitionModel, self).__init__()
        
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = vmodels.r2plus1d_18(weights=weights)
        
        if freeze_early_layers and pretrained:
            # Freeze stem and first 2 residual blocks to preserve low-level motion features
            for param in self.backbone.stem.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
                
        # Replace 400-class head with our 7-class surveillance action head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Expected input shape: (Batch, Channels=3, Time=16, Height=112, Width=112)
        return self.backbone(x)

class MobileNetV3_GRU(nn.Module):
    """
    Ultra-lightweight fallback architecture:
    MobileNetV3-Large 2D CNN feature extractor + Bidirectional GRU temporal head.
    Suitable for devices with constrained memory.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, hidden_dim=256):
        super(MobileNetV3_GRU, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        feature_dim = 960
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Reshape to (B*T, C, H, W) for 2D CNN
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = self.pool(features).view(B, T, -1) # Shape: (B, T, 960)
        
        gru_out, _ = self.gru(features)
        # Global temporal pooling (mean over time)
        temporal_pooled = torch.mean(gru_out, dim=1)
        out = self.classifier(temporal_pooled)
        return out

def build_model(model_name="r2plus1d_18", num_classes=NUM_CLASSES, pretrained=True, freeze_early_layers=False):
    """
    Model factory to initialize action recognition models.
    """
    if model_name == "r2plus1d_18":
        return ActionRecognitionModel(num_classes=num_classes, pretrained=pretrained, freeze_early_layers=freeze_early_layers)
    elif model_name == "mobilenet_gru":
        return MobileNetV3_GRU(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
