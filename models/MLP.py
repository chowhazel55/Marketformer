import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self, seq_len=12, pred_len=12, d_feat=5 , d_model=256,dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_feat = d_feat
        self.d_model = d_model

        self.mlp = nn.Sequential(
            nn.Linear(seq_len * d_feat, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        # x: ( N,T,C)
        num_nodes = x.shape[0]

        x = x.transpose(1, 2).reshape(num_nodes, -1)  # (N, T*C)

        out = self.mlp(x)  # (B, N, out_steps*output_dim)

        return out.squeeze()