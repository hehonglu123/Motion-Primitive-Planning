import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from replayer import Replayer


class FeatureNetFCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(FeatureNetFCN, self).__init__()
        self.output_dim = output_dim
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


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(FCN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class DQNPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, feature_net: FeatureNetFCN = None, hidden_dim: int = 64):
        super(DQNPolicy, self).__init__()
        self.feature_net = feature_net
        self.input_dim = input_dim + feature_net.output_dim if feature_net is not None else input_dim
        self.linear1 = nn.Linear(self.input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x, curve=None):
        if self.feature_net is not None:
            if curve is None:
                raise Exception("DQNPolicy: Missing input curve.")
            curve_feature = self.feature_net(curve)
            x = torch.cat([curve_feature, x], dim=1)
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


class DQNAgent(object):
    def __init__(self, args):
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma

        self.feature = FeatureNetFCN(args.curve_dim, args.action_dim)
        self.policy = DQNPolicy(input_dim=args.feature_dim, output_dim=args.action_dim, feature_net=self.feature)
        self.policy_target = copy.deepcopy(self.policy)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    def evaluate(self, state):
        self.policy.eval()
        with torch.no_grad():
            curve, primitive_feature = state
            curve_tensor = torch.tensor(curve).flatten().unsqueeze(0)
            feature_tensor = torch.tensor(primitive_feature).flatten().unsqueeze(0)
            action_probs = self.policy(feature_tensor, curve_tensor).cpu().numpy().flatten()
            action = np.argmax(action_probs)
            return action

    def choose_action(self, state):
        self.policy.eval()
        with torch.no_grad():
            curve, primitive_feature = state
            curve_tensor = torch.tensor(curve).flatten().unsqueeze(0)
            feature_tensor = torch.tensor(primitive_feature).flatten().unsqueeze(0)
            action_probs = self.policy(feature_tensor, curve_tensor)
            dist = Categorical(probs=action_probs)
            action = dist.sample()
            return action.cpu().numpy()[0]

    def learn(self, replayer: Replayer):
        state_curve, state_feature, action, reward, next_state_curve, next_state_feature, done = replayer.sample(self.batch_size)
        self.policy.train()

        with torch.no_grad():
            target_out = self.policy_target(next_state_feature, next_state_curve)
            q_target = reward + self.gamma * (1 - done) * target_out.max(1, keepdim=True)[0]

        q_estimate = self.policy_target(state_feature, state_curve).gather(1, action)
        loss = self.loss_fn(q_estimate, q_target)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path + os.sep + 'DQN_policy.pth')
        torch.save(self.policy_target.state_dict(), path + os.sep + 'DQN_target.pth')

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path + os.sep + 'DQN_policy.pth'))
        self.policy_target.load_state_dict(torch.load(path + os.sep + 'DQN_target.pth'))


