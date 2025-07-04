from block import Block
import torch.nn as nn
from torch.nn.functional import cross_entropy,softmax
from positionalembedding import positional_embedding
import torch

class Transformer(nn.Module):
    def __init__(self,vocab_size,emb_size,max_seq,num_head,num_block):
        super().__init__()
        self.context_len = max_seq
        self.token_emb = nn.Embedding(vocab_size,emb_size)
        pe = positional_embedding(max_seq,emb_size).pos_encode()
        self.register_buffer("pos_emb",pe)
        #self.pos_emb = nn.Embedding(max_seq,emb_size)
        self.block = nn.ModuleList([Block(num_head,emb_size,max_seq) for _ in range(num_block)])
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.2)
        self.lm = nn.Linear(emb_size,vocab_size)
    def block_inp(self,x):
        B,T = x.shape
        tok_emb = self.token_emb(x)
        #pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        pos_emb = self.pos_emb[:T, :].unsqueeze(0)
        return self.dropout(tok_emb + pos_emb)
    def forward(self,x,targets=None):
        x = self.block_inp(x)
        for B in self.block:
            x = B(x) 
        x = self.norm1(x)
        logits = self.lm(x)
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = cross_entropy(logits,targets)
        return logits,loss
    def generate(self,x,max_tokens,temp=1.0):
        for _ in range(max_tokens):
            idx = x[:,-self.context_len:]
            logit,_ = self(idx)
            logit = logit[:,-1,:]
            logit = logit / temp
            probs = softmax(logit,dim=-1)
            out = torch.multinomial(probs,num_samples=1)
            x = torch.cat([x,out],dim=-1)
        return x
            
        
        
        