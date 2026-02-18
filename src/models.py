import torch.nn as nn
import torch.nn.functional as F
import timm

class SimCLR(nn.Module):
    def __init__(self, backbone="resnet18", proj_dim=128):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat_dim = self.encoder.num_features
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1)
        return h, z
