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

        self.attention_dropout = nn.Dropout(p=attention_pdropout)
        self.output_dropout = nn.Dropout(p=output_pdropout)

        self.Wq = nn.Linear(k, k, bias=False)
        self.Wk = nn.Linear(k, k, bias=False)
        self.Wv = nn.Linear(k, k, bias=False)
        self.Wo = nn.Linear(k, k, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_length + 1, max_length + 1)).view(1, 1, max_length + 1, max_length + 1)
        )


    def forward(self, x):
        b, t, k = x.size()
        assert k == self.k
        hs = self.k // self.n_heads
        Q = self.Wq(x).view(b, t, self.n_heads, hs).transpose(1, 2) 
        K = self.Wk(x).view(b, t, self.n_heads, hs).transpose(1, 2) 
        V = self.Wv(x).view(b, t, self.n_heads, hs).transpose(1, 2) 
        W = (Q @ K.transpose(-2, -1)) / (k ** .5)
        mask = self.mask[:, :, :t, :t]
        W = W.masked_fill(mask == 0, float("-inf"))
        W = F.softmax(W, dim=-1)
        W = self.attention_dropout(W)
        O = W @ V
        O = O.transpose(1, 2).contiguous().view(b, t, k)
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

        self.time_embed = nn.Sequential(
            nn.Linear(1, embedding_dim), 
            nn.Tanh()
        ).to(device=device)

        self.all_pos_embed = nn.Parameter(torch.zeros((1, 3 * max_length, embedding_dim), device=device))
        
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


    def configure_optimizers(self, lr, weight_decay, betas):
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('all_pos_embed')
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer


class DecisionTransformer(TransformerModel):

    def forward(self, states, actions, rtgs, timesteps=None):
        batch_size, length  = states.size()
        pos_embeddings = self.time_embed(timesteps.unsqueeze(-1))
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        rtg_embeddings = self.return_embed(rtgs.unsqueeze(-1))
        input_embeddings = torch.zeros((batch_size, 3 * length, self.embedding_dim), 
                                        dtype=torch.float32, device=self.device)
        input_embeddings[:, 0::3, :] = rtg_embeddings + pos_embeddings
        input_embeddings[:, 1::3, :] = state_embeddings + pos_embeddings
        input_embeddings[:, 2::3, :] = action_embeddings + pos_embeddings
        position_embeddings = self.all_pos_embed[:, :3 * length, :]
        hidden_states = self.embedding_dropout(input_embeddings + position_embeddings)
        hidden_states = self.blocks(hidden_states)
        hidden_states = self.norm(hidden_states)
        action_hidden = hidden_states[:, 2::3, :]
        logits = self.fc(action_hidden.reshape(-1, self.embedding_dim)).reshape(batch_size, length, -1)
        return logits


class BehaviorCloning(TransformerModel):

    def forward(self, states, actions, rtgs=None, timesteps=None):
        batch_size, length = states.size()
        pos_embeddings = self.time_embed(timesteps.unsqueeze(-1))
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        input_embeddings = torch.zeros((batch_size, 2 * length, self.embedding_dim), 
                                        dtype=torch.float32, device=self.device)
        input_embeddings[:, 0::2, :] = state_embeddings + pos_embeddings
        input_embeddings[:, 1::2, :] = action_embeddings + pos_embeddings
        position_embeddings = self.all_pos_embed[:, :2 * length, :]
        hidden_states = self.embedding_dropout(input_embeddings + position_embeddings)
        hidden_states = self.blocks(hidden_states)
        hidden_states = self.norm(hidden_states)
        action_hidden = hidden_states[:, 1::2, :]
        logits = self.fc(action_hidden.reshape(-1, self.embedding_dim)).reshape(batch_size, length, -1)
        return logits
