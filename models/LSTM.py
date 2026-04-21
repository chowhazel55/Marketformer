import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self,d_feat=158, hidden_size=256, num_layers=2, dropout=0.0,gate_input_start_index=158,gate_input_end_index=221):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index #158
        self.gate_input_end_index = gate_input_end_index #221
        self.rnn=nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,      
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out=nn.Linear(hidden_size,1)
        self.d_feat=d_feat
    
    def forward(self, x):
        #src = x[:, :, :self.gate_input_start_index]
        if (self.d_feat==158):
            x = x[:, :, :self.gate_input_start_index]
            #src = src[:, :,[1,5,17,23,33,34,35,37,38,39,88,89,92,93,94,95,97,133,138,142]] # B, T, D 300*T*158
        out,_=self.rnn(x.squeeze(dim=0))
        return self.fc_out(out[:,-1,:]).squeeze(dim=1)