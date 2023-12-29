# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, k, n_heads, attention_pdropout=0.0, output_pdropout=0.0, max_length=768):
        super().__init__()
        assert k % n_heads == 0
        self.k = k
        self.n_heads = n_heads

        self.mask = torch.tril(torch.ones(max_length + 1, max_length + 1)).view(1, 1, max_length + 1, max_length + 1)

        self.attention_dropout = nn.Dropout(p=attention_pdropout)
        self.output_dropout = nn.Dropout(p=output_pdropout)

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
        mask = self.mask[:, :, :t, :t]
        W = W.masked_fill(mask == 0, float("-inf")).squeeze(0)
        W = F.softmax(W, dim=2)
        W = self.attention_dropout(W)

        O = torch.bmm(W, V).view(b, self.n_heads, t, hs)
        O = O.transpose(1, 2).contiguous().view(b, t, self.k)
        y = self.Wo(O)
        y = self.output_dropout(y)

        return y


class GeLU(nn.Module):

    def forward(self, x):
        return F.gelu(x)


class TransformerBlock(nn.Module):

    def __init__(self, k, n_heads, attention_pdropout=0.0, output_pdropout=0.0, max_length=768):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        
        self.attention = MultiHeadAttention(k, n_heads, attention_pdropout, output_pdropout, max_length)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.feed_forward = nn.Sequential(
            nn.Linear(k, 4 * k),
            GeLU(),
            nn.Linear(4 * k, k),
            nn.Dropout(p=output_pdropout)
        )

    
    def forward(self, x):
        x = self.norm1(self.attention(x) + x)
        x = self.norm2(self.feed_forward(x) + x)
        return x
        

# TODO:
class DecisionTransformer(nn.Module):

    def __init__(self,
                vocab_size,
                max_length,

                n_heads=12,
                n_blocks=12,
                embedding_dim=768,
                attention_pdropout=0.0,
                embedding_pdropout=0.0,
                output_pdropout=0.0,

                device=None
                ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        self.state_embed = None
        self.actions_embed = None
        
        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device=device)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embedding_dim), device=device)
        self.embedding_dropout = nn.Dropout(p=embedding_pdropout)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)


    def forward(self, x, target=None):
        b, t, k = x.size()
        token_embeddings = token_embeddings(x)
        pos_embeddings = self.pos_embed[:, :t, :]
        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.norm(x)
        x = self.fc(x)
        return x
