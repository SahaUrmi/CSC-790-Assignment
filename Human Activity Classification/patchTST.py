import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class HAR_PatchTST(nn.Module):
    def __init__(self, input_dim=6, seq_len=128, patch_size=16, d_model=64, nhead=4, num_layers=2, num_classes=12, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.input_dim = input_dim
        self.d_model = d_model

        assert seq_len % patch_size == 0, "Sequence length must be divisible by patch size."

        # Patch embedding: flatten (patch_size, input_dim) → d_model
        self.patch_embed = nn.Linear(patch_size * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) where T = seq_len, D = input_dim
        Returns:
            logits: (B, num_classes)
        """
        B, T, D = x.shape

        # Step 1: Patchify: (B, T, D) → (B, num_patches, patch_size * D)
        x = x.view(B, self.num_patches, self.patch_size * D)

        # Step 2: Project each patch
        x = self.patch_embed(x) + self.pos_embed  # (B, N, d_model)

        # Step 3: Transformer
        x = self.encoder(x)  # (B, N, d_model)

        # Step 4: Pool over tokens
        x = x.permute(0, 2, 1)  # (B, d_model, N)
        x = self.pool(x)        # (B, d_model, 1)

        # Step 5: Classification
        return self.classifier(x)  # (B, num_classes)
