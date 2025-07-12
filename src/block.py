import torch.nn as nn
from .feedforward import MLP
from .attention import Attention,MultiHeadAttention

class Block(nn.Module):
    def __init__(self, num_head, emb_size, context):
        super().__init__()
        self.head_size = emb_size//num_head
        self.mha = MultiHeadAttention(num_head,self.head_size,emb_size,context)
        self.FF = MLP(emb_size)
        self.l1 = nn.LayerNorm(emb_size)
        self.l2 = nn.LayerNorm(emb_size)
    def forward(self,x):
        x = x + self.mha(self.l1(x))
        x = x + self.FF(self.l2(x))
        return x