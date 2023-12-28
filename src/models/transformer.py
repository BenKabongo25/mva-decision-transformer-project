# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, k, n_heads, mask=False):
        super().__init__()
        assert k % n_heads == 0
        self.k = k
        self.n_heads = n_heads
        self.mask = mask

        self.Wq = nn.Linear(k, k, bias=False)
        self.Wk = nn.Linear(k, k, bias=False)
        self.Wv = nn.Linear(k, k, bias=False)
        self.Wo = nn.Linear(k, k, bias=False)


    def forward(self, x):
        b, t, k = x.size()
        assert k == self.k
        hs = self.k // self.n_heads

        Q = self.Wq(x).view(b, t, self.n_heads, hs)
        K = self.Wk(x).view(b, t, self.n_heads, hs)
        V = self.Wv(x).view(b, t, self.n_heads, hs)

        Q = Q.transpose(1, 2).contiguous().view(b * self.n_heads, t, hs)
        K = K.transpose(1, 2).contiguous().view(b * self.n_heads, t, hs)
        V = V.transpose(1, 2).contiguous().view(b * self.n_heads, t, hs)

        W = torch.bmm(Q, K.transpose(1, 2)) / (self.k ** .5)
        # TODO: mask
        W = F.softmax(W, dim=2)

        O = torch.bmm(W, V).view(b, self.n_heads, t, hs)
        O = O.transpose(1, 2).contiguous().view(b, t, self.k)
        y = self.Wo(O)

        return y


class TransformerBlock(nn.Module):

    def __init__(self, k, n_heads, mask=False):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        
        self.attention_layer = MultiHeadAttention(k, n_heads, mask)
        self.layer_norm1 = nn.LayerNorm(k)
        self.layer_norm2 = nn.LayerNorm(k)
        self.feed_forward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    
    def forward(self, x):
        x = self.layer_norm1(self.attention_layer(x) + x)
        x = self.layer_norm2(self.feed_forward(x) + x)
        return x
        

