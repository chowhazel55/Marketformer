from torch import nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=158, d_model=256, nhead=4, num_layers=1, dropout=0, gate_input_start_index=158):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.d_feat = d_feat
        self.gate_input_start_index=gate_input_start_index

    def forward(self, src):
        src = src[:, :, :self.d_feat]
        src = self.feature_layer(src)  # [512, 60, 8]
        src = src.transpose(1, 0)  # not batch first
        mask = None
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]
        return output.squeeze(dim=1)