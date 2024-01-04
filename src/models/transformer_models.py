# Deep Learning
# January 2024
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


class TransformerModel(nn.Module):

    def __init__(self,
                n_states,
                n_actions,
                max_length,
                max_t,
                n_heads=12,
                n_blocks=12,
                embedding_dim=768,
                attention_pdropout=0.0,
                embedding_pdropout=0.0,
                output_pdropout=0.0,
                device=None
                ):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.max_length = max_length
        self.max_t = max_t
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.embedding_dim = embedding_dim
        self.device = device

        self.embedding_dropout = nn.Dropout(p=embedding_pdropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, n_heads, attention_pdropout, output_pdropout, max_length) 
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, n_actions)

        self.state_embed = nn.Sequential(
            nn.Embedding(num_embeddings=n_states, embedding_dim=embedding_dim),
            nn.Tanh()
        ).to(device=device)

        self.return_embed = nn.Sequential(
            nn.Linear(1, embedding_dim), 
            nn.Tanh()
        ).to(device=device)

        self.action_embed = nn.Sequential(
            nn.Embedding(num_embeddings=n_actions, embedding_dim=embedding_dim), 
            nn.Tanh()
        ).to(device=device)

        self.pos_embed = nn.Parameter(torch.zeros((1, max_length, embedding_dim), device=device))
        self.all_pos_embed = nn.Parameter(torch.zeros((1, max_t, embedding_dim), device=device))
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, states, actions=None, rtgs=None, timesteps=None):
        raise NotImplementedError


class DecisionTransformer(TransformerModel):

    def forward(self, states, actions=None, rtgs=None, timesteps=None):
        batch_size, length, _ = states.size()

        state_embeddings = self.state_embed(states.type(torch.long).squeeze(-1))
        pos_embeddings = self.pos_embed[:, :length, :]
        
        if actions is not None: 
            action_embeddings = self.action_embed(actions.type(torch.long).squeeze(-1))
            rtg_embeddings = self.return_embed(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((batch_size, length * 3, self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::3, :] = rtg_embeddings + pos_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings + pos_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings + pos_embeddings

        else:
            rtg_embeddings = self.return_embed(rtgs.type(torch.float32)) 
            token_embeddings = torch.zeros((batch_size, length * 2, self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::2, :] = rtg_embeddings + pos_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings + pos_embeddings

        position_embeddings = self.all_pos_embed[:, :token_embeddings.shape[1], :]

        x = self.embedding_dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.fc(x)
        logits = logits[:, 1::3, :] if actions is not None else logits[:, 1:, :]
        return logits


class BehaviorCloning(TransformerModel):

    def forward(self, states, actions=None, rtgs=None, timesteps=None):
        batch_size, length, _ = states.size()

        state_embeddings = self.state_embed(states.type(torch.long).squeeze(-1))
        pos_embeddings = self.pos_embed[:, :length, :]

        if actions is not None:
            action_embeddings = self.action_embed(actions.type(torch.long).squeeze(-1))
            token_embeddings = torch.zeros((batch_size, length * 2, self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::2, :] = state_embeddings + pos_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings + pos_embeddings
        else:
            token_embeddings = state_embeddings + pos_embeddings

        position_embeddings = self.all_pos_embed[:, :token_embeddings.shape[1], :]
        x = self.embedding_dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.fc(x)
        logits = logits[:, 0::2, :] if actions is not None else logits
        return logits
