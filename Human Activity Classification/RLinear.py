import torch
import torch.nn as nn

class RLinearClassifier(nn.Module):
    def __init__(self, input_dim, proj_dim, num_classes):
        super().__init__()
        self.linear_proj = nn.Linear(input_dim, proj_dim)   # input_dim = 128
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, x):  # x: [B, T, C] = [B, 128, 9]
        x = x.transpose(1, 2)         # [B, C, T] = [B, 9, 128]
        x = self.linear_proj(x)       # [B, C, D] = [B, 9, 64]
        x = x.mean(dim=1)             # [B, D] = [B, 64]
        return self.classifier(x)     # [B, num_classes]
