# Deep Learning
# December 2023
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.device = device

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

        self.encoder = nn.GRU(3 * embedding_dim, hidden_size, batch_first=True, device=device)
        self.project_fc = nn.Linear(hidden_size, project_size)
        self.out_fc = nn.Linear(project_size, n_actions)
        self.relu = nn.ReLU()


    def forward(self, states, actions, rtgs, timesteps=None):
        batch_size, length = states.size()
        state_embeddings  = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        rtg_embeddings = self.return_embed(rtgs.unsqueeze(-1))
        embeddings = torch.zeros((batch_size, 3 * length, self.embedding_dim), 
                                dtype=torch.float32, device=self.device)
        embeddings[:, 0::3, :] = rtg_embeddings
        embeddings[:, 1::3, :] = state_embeddings
        embeddings[:, 2::3, :] = action_embeddings
        embeddings = embeddings.reshape((batch_size, length, 3 * self.embedding_dim))
        outputs, _ = self.encoder(embeddings)
        return self.out_fc(self.relu(self.project_fc(outputs)))
