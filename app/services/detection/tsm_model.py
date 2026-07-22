"""
Temporal Shift Module (TSM) architecture for KAVACH AI 2.0 Deepfake Detection.
Grants 3D-CNN temporal awareness to 2D-CNNs without extra computational cost.
"""

import torch
import torch.nn as nn
from torchvision import models

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8):
        # x shape: (B*T, C, H, W)
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)
        # Shift left (past frame)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # Shift right (future frame)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        # No shift (current frame)
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)

class TSMEfficientNet(nn.Module):
    def __init__(self, num_classes=1, num_segments=8, pretrained=True):
        super(TSMEfficientNet, self).__init__()
        self.num_segments = num_segments
        
        # Load base 2D model
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = models.efficientnet_b0(weights=weights)
        
        # Inject Temporal Shift Modules into the MBConv blocks
        self._insert_tsm()
        
        # Replace classifier head for our task
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def _insert_tsm(self):
        """
        Injects TSM before the residual blocks in EfficientNet's features.
        In EfficientNet, features[1:] are MBConv blocks.
        """
        for i in range(1, len(self.base_model.features) - 1):
            block = self.base_model.features[i]
            # Wrap the entire block's first layer with TSM
            self.base_model.features[i] = TemporalShift(block, n_segment=self.num_segments, n_div=8)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        b, t, c, h, w = x.size()
        
        if t != self.num_segments:
            raise ValueError(f"Expected sequence of {self.num_segments} frames, got {t}")
            
        # Reshape to (B*T, C, H, W) for 2D backbone
        x = x.view(b * t, c, h, w)
        
        # Pass through TSM-enabled network
        x = self.base_model(x)
        
        # x is now (B*T, num_classes)
        x = x.view(b, t, -1)
        
        # Temporal Consensus (Average pooling across time)
        out = torch.mean(x, dim=1)
        
        return out
