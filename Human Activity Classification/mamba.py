import torch.nn as nn
from mamba_ssm import Mamba

class HAR_Mamba(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_classes):
                super().__init__()
                self.proj_in = nn.Linear(input_dim, hidden_dim)
                self.mamba = Mamba(d_model=hidden_dim)
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),      # pool over time
                    nn.Flatten(),
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.Linear(hidden_dim//2, num_classes)
                )

            def forward(self, x):
                # x: (batch_size, seq_len, input_dim)
                x = self.proj_in(x)               # (batch_size, seq_len, hidden_dim)
                x = self.mamba(x)                 # (batch_size, seq_len, hidden_dim)
                x = x.transpose(1, 2)             # for pooling: (batch_size, hidden_dim, seq_len)
                return self.classifier(x)         # (batch_size, num_classes)