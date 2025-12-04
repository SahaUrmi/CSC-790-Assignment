import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class HAR_iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(HAR_iTransformer, self).__init__()
        self.seq_len = configs.window_size
        # self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.window_size, configs.emb_dim)
        
        # self.class_strategy = configs.class_strategy
        
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(), configs.emb_dim, configs.n_heads),
                    configs.emb_dim
                    # configs.d_ff,
                    # dropout=configs.dropout,
                    # activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.emb_dim)
        )
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),      # pool over time
                    nn.Flatten(),
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.Linear(hidden_dim//2, num_classes)
                )

    def forecast(self, x_enc):

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: emb_dim; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        return enc_out, attns


    def forward(self, x_enc, mask=None):
        out, _ = self.forecast(x_enc)
        print(f"Output from iTransformer blocks: {out.shape}")

        return out
        