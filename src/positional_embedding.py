import torch
import math

class positional_embedding():
    def __init__(self,seq,d_size):
        self.seq_len = seq
        self.dim = d_size
    def pos_encode(self):
        """
        Computes the positional encoding matrix as described in the original Transformer paper:

            PE(pos, 2i)   = sin(pos / (10000^(2i / d_model)))
            PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))

        Args:
            seq (int)              : Length of the input sequence.
            d_model (int)          : Dimension of the model.

        Returns:
            torch.Tensor: A tensor of shape (seq, d_model) containing positional encodings.
        """
        pos_emb = torch.zeros(self.seq_len,self.dim)
        sine = lambda p,i: math.sin(p/10000**((2*i)/self.dim))
        cosine = lambda p,i: math.cos(p/10000**((2*i)/self.dim))
        for k in range(self.seq_len):
            for j in range(0,self.dim,2):
                    pos_emb[k,j] = sine(k,j)
                    pos_emb[k,j+1] = cosine(k,j+1)
        self.pe = pos_emb
        return self.pe