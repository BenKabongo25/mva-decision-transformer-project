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


class TransformerModel(nn.Module):

    def __init__(self,
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

        self.n_actions = n_actions
        self.max_length = max_length
        self.max_t = max_t
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.token_embed = nn.Embedding(num_embeddings=n_actions, embedding_dim=embedding_dim, device=device)
        self.pos_embed = nn.Parameter(torch.zeros((1, max_length, embedding_dim), device=device))
        self.all_pos_embed = nn.Parameter(torch.zeros((1, max_t, embedding_dim), device=device))
        self.embedding_dropout = nn.Dropout(p=embedding_pdropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, n_heads, attention_pdropout, output_pdropout, max_length) 
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, n_actions)

        self.apply(self._init_weights)

        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(3136, embedding_dim), 
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
        nn.init.normal_(self.action_embed[0].weight, mean=0.0, std=0.02)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, states, actions=None, targets=None, rtgs=None, timesteps=None):
        raise NotImplementedError


    def configure_optimizers(self, lr, weight_decay, betas):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
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

        no_decay.add('pos_embed')
        no_decay.add('all_pos_embed')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer


class DecisionTransformer(TransformerModel):

    def forward(self, states, actions=None, targets=None, rtgs=None, timesteps=None):
        batch_size, length, _ = states.size()

        state_embeddings = self.state_encoder(
            states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
            ).reshape(batch_size, length, self.embedding_dim)
        
        if actions is not None: 
            action_embeddings = self.action_embed(actions.type(torch.long).squeeze(-1))
            rtg_embeddings = self.return_embed(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((batch_size, length * 3 - int(targets is None), self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -length + int(targets is None):, :]

        else:
            rtg_embeddings = self.return_embed(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((batch_size, length * 2, self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::2, :] = rtg_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings

        all_pos_embeddings = torch.repeat_interleave(self.all_pos_embed, batch_size, dim=0)
        position_embeddings = torch.gather(
            all_pos_embeddings, 1, 
            torch.repeat_interleave(timesteps, self.embedding_dim, dim=-1)
        ) 
        print(position_embeddings.size())
        position_embeddings += self.pos_embed[:, :token_embeddings.shape[1], :]

        x = self.embedding_dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.fc(x)
        logits = logits[:, 1::3, :] if actions is not None else logits[:, 1:, :]
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class BehaviorCloning(TransformerModel):

    def forward(self, states, actions=None, targets=None, rtgs=None, timesteps=None):
        batch_size, length, _ = states.size()

        state_embeddings = self.state_encoder(
            states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()
            ).reshape(batch_size, length, self.embedding_dim)

        if actions is not None:
            action_embeddings = self.action_embed(actions.type(torch.long).squeeze(-1))
            token_embeddings = torch.zeros((batch_size, length * 2 - int(targets is None), self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -length + int(targets is None):, :]
        else:
            token_embeddings = state_embeddings

        all_pos_embeddings = torch.repeat_interleave(self.all_pos_embed, batch_size, dim=0)
        position_embeddings = torch.gather(
            all_pos_embeddings, 1, 
            torch.repeat_interleave(timesteps, self.embedding_dim, dim=-1)
        )
        position_embeddings += self.pos_embed[:, :token_embeddings.shape[1], :]

        x = self.embedding_dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.fc(x)
        logits = logits[:, 0::2, :] if actions is not None else logits
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
