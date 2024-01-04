# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, device=None):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, device=device)

    def forward(self, input):
        output, hn = self.rnn(input)
        return output, hn


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, project_size, output_size, device=None):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, device=device)
        self.project_fc = nn.Linear(hidden_size, project_size)
        self.out_fc = nn.Linear(project_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input, h0):
        output, hn = self.rnn(input, h0)
        output = self.out_fc(self.relu(self.project_fc(output)))
        return output, hn


class RNN4RL(nn.Module):

    def __init__(
        self,
        n_states,
        n_actions,
        embedding_dim,
        hidden_size,
        project_size,
        embedding_pdropout=0.0,
        device=None
        ):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim

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

        self.dropout = nn.Dropout(p=embedding_pdropout)

        self.encoder = Encoder(input_size=3 * embedding_dim, hidden_size=hidden_size, device=device)
        self.decoder = Decoder(input_size=hidden_size, hidden_size=hidden_size, project_size=project_size, 
                               output_size=n_actions, device=device)

    def forward(self, states, actions=None, rtgs=None, timesteps=None):
        batch_size, length, _ = states.size()

        state_embeddings  = self.state_embed(states.type(torch.long).squeeze(-1))

        if actions is not None: 
            action_embeddings = self.action_embed(actions.type(torch.long).squeeze(-1))
            rtg_embeddings = self.return_embed(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((batch_size, length * 3, self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings

        else:
            rtg_embeddings = self.return_embed(rtgs.type(torch.float32)) 
            token_embeddings = torch.zeros((batch_size, length * 2, self.embedding_dim), 
                                            dtype=torch.float32, device=self.device)
            token_embeddings[:, 0::2, :] = rtg_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings
        
        