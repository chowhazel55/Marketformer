import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketStock_CrossAttention(nn.Module):
    def __init__(self,n_node,d_emb,d_model,nheads,alpha):
        super().__init__()
        self.alpha=alpha
        self.emb = nn.Parameter(torch.randn(n_node+1,d_emb))
        self.We = nn.Parameter(torch.randn(d_emb,d_model,d_model))
        self.be = nn.Parameter(torch.randn(d_emb,d_model))
        self.nheads=nheads
        self.scale=(d_model//nheads)**0.5
        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def adjacencyMatrix(self,device):
        diff = self.emb.unsqueeze(1)-self.emb.unsqueeze(0)
        sim = torch.sum(diff,dim=-1)
        A = F.softmax(F.elu(sim,alpha=self.alpha),dim=-1)
        A=A+torch.eye(A.shape[0],device=device)
        return A
    
    def updateQ(self,q,A):
        q=q.transpose(0,1) #[T,N,D]
        W = torch.einsum('ne,eio->nio',self.emb,self.We)
        b = torch.einsum('ne,eo->no',self.emb,self.be)
        q_ = torch.einsum('nm,tmd->tnd',A,q)
        q_ = torch.einsum('tnd,ndo->tno',q_,W)+b
        return q_.transpose(0,1)
    
    def forward(self,q,k,v):
        # N+1, T, D
        B,T,D = q.shape
        device=q.device
        q=self.q_linear(q)
        k=self.k_linear(k)
        v=self.v_linear(v)    
        A=self.adjacencyMatrix(device)
        q=self.updateQ(q,A)
        q = torch.cat(torch.split(q, D//self.nheads, dim=-1), dim=0)
        k = torch.cat(torch.split(k, D//self.nheads, dim=-1), dim=0)
        v = torch.cat(torch.split(v, D//self.nheads, dim=-1), dim=0)
        attn_score = torch.softmax((q @ k.transpose(-1,-2)) / self.scale, dim=-1)
        out = attn_score @ v
        out = torch.cat(torch.split(out, B, dim=0), dim=-1) 
        return self.out_proj(out)

class SelfAttentionLayer(nn.Module):
    def __init__(self,n_node,d_emb,d_model,nheads,alpha, dff,dropout):
        super().__init__()
        self.msAttention = MarketStock_CrossAttention(n_node,d_emb,d_model,nheads,alpha)
        self.ffc = nn.Sequential(
            nn.Linear(d_model,dff),
            nn.ReLU(inplace=True),
            nn.Linear(dff,d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self,x):
        # x=x.transpose(dim,-2)
        bias = x
        out=self.msAttention(x,x,x)
        out=self.dropout1(out)
        out=self.ln1(bias+out)
        bias = out
        out=self.ffc(out)
        out=self.dropout2(out)
        out=self.ln2(bias+out)
        return out #.transpose(dim,-2)

    
class Marketformer(nn.Module):
    def __init__(self,d_feat=5,seq_len=16,pred_len=1,n_node=474,d_emb=24,d_model=64,dff=256,nheads=2,num_layers=2,alpha=1,dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_feat,d_model)
        self.virtual_market_generator = nn.Sequential(
            nn.Linear(n_node, dff),
            nn.ReLU(inplace = True),
            nn.Linear(dff,1),
        )
        self.virtual_market_transformer = nn.ModuleList([
            SelfAttentionLayer(n_node,d_emb,d_model,nheads,alpha, dff,dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(seq_len *d_model,pred_len)
    def forward(self,x):
        x=self.input_proj(x)
        virutal_market = self.virtual_market_generator(x.permute(1,2,0)).permute(2,0,1)
        x = torch.cat([x,virutal_market],dim=0) #N+1,T,D
        for layer in self.virtual_market_transformer:
            x=layer(x)
        out=self.out_proj(x.reshape(x.shape[0],-1))
        return out.squeeze()[:-1]

        
