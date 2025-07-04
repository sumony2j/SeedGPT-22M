import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.layer = nn.Linear(emb_size,4*emb_size)
        self.activation = nn.ReLU()
        self.proj = nn.Linear(4*emb_size,emb_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        x = self.forward_map(x)
        x = self.forward_proj(x)
        return x
    def forward_map(self,x):
        x = self.activation(self.layer(x))
        return x
    def forward_proj(self,x):
        x = self.proj(x)
        x = self.dropout(x)
        return x
