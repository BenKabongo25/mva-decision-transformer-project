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


class Seq2SeqDiscreteAction(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_size,
        project_size,
        n_actions,
        state_dim=1,
        embedding_pdropout=0.0,
        device=None
        ):
        super().__init__()

        self.state_embed  = nn.Linear(state_dim, embedding_dim, device=device)
        self.action_embed = nn.Embedding(n_actions, embedding_dim, device=device)
        self.reward_embed = nn.Linear(1, embedding_dim, device=device)
        self.dropout = nn.Dropout(p=embedding_pdropout)

        self.encoder = Encoder(input_size=3 * embedding_dim, hidden_size=hidden_size, device=device)
        self.decoder = Decoder(input_size=hidden_size, hidden_size=hidden_size, project_size=project_size, 
                               output_size=n_actions, device=device)

    def forward(self, states, actions, rewards):
        state_embeddings  = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        reward_embeddings = self.reward_embed(rewards)

        embeddings = torch.stack([reward_embeddings, state_embeddings, action_embeddings])
        embeddings = self.dropout(embeddings)

        _, hiddens = self.encoder(embeddings)
        # TODO: None
        last_state  = states[:, -1, :].unsqueeze(1)
        last_action = actions[:, -1, :].unsqueeze(1)
        last_reward = rewards[:, -1, :].unsqueeze(1)
        decoder_input = #
        