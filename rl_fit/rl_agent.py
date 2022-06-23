import torch
import torch.nn as nn


class FeatureNetFCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(FeatureNetFCN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftActorCriticAgent(object):
    def __init__(self, args):
        self.feature_net = FeatureNetFCN(args.curve_dim, args.action_dim)
        self.actor = Actor(args.feature_dim + args.action_dim, args.action_dim)
        self.critic = Critic(args.feature_dim + args.action_dim, args.action_dim)
