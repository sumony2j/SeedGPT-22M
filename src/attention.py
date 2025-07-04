import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self,emb_size,head_size,context):
        super().__init__()
        self.Q = nn.Linear(emb_size,head_size,bias=False)
        self.K = nn.Linear(emb_size,head_size,bias=False)
        self.V = nn.Linear(emb_size,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(context,context)))
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        B,T,C = x.shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        wei = q @ k.transpose(-2,-1) * C ** -0.05
        wei = wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei = torch.softmax(wei,dim=-1)
        out = wei @ v
        out = self.dropout(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head,head_size,emb_size,context):
        super().__init__()
        self.mha = nn.ModuleList([Attention(emb_size=emb_size,head_size=head_size,
                                             context=context) for _ in range(num_head)])
        self.proj = nn.Linear(num_head*head_size,emb_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.mha],dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
        
        
        
         