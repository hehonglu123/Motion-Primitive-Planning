import copy
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveParmNoise(object):
    def __init__(self, init_std=0.2, target_std=0.1, adapt_coefficient=1.01):
        self.init_std = init_std
        self.target_std = target_std
        self.adapt_coefficient = adapt_coefficient
        self.current_std = self.init_std

    def adapt(self, distance):
        if distance > self.target_std:
            self.current_std /= self.adapt_coefficient
        else:
            self.current_std *= self.adapt_coefficient

    def get_noisy_model(self, model: nn.Module):
        with torch.no_grad():
            new_model = copy.deepcopy(model)
            for name, params in new_model.named_parameters():
                params.data.copy_(params.data + torch.normal(mean=torch.zeros(params.shape),
                                                             std=torch.ones(params.shape) * self.current_std))
            return new_model


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, action_dim, mu=0., sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu * np.ones(action_dim)
        self.sigma = sigma * np.ones(action_dim)
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class GaussianActionNoise(object):
    def __init__(self, action_dim, mu=0, sigma=0.3):
        self.dim = action_dim
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma, size=self.dim)


class Feature(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Feature, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x


class Actor(nn.Module):
    def __init__(self, feature_net: Feature, input_dim, action_dim, max_action, hidden_dim=64):
        super(Actor, self).__init__()

        self.max_action = max_action
        self.feature_net = feature_net
        self.input_dim = input_dim + self.feature_net.output_dim
        self.l1 = nn.Linear(self.input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()

    def forward(self, curve, robot):
        curve_feature = self.feature_net(curve)
        x = torch.cat([curve_feature, robot], dim=1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        action = self.max_action * torch.tanh(x)
        return action


class Critic(nn.Module):
    def __init__(self, feature_net: Feature, input_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()

        self.feature_net = feature_net
        self.input_dim = self.feature_net.output_dim + input_dim + action_dim
        self.q1_l1 = nn.Linear(self.input_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)

        self.q2_l1 = nn.Linear(self.input_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, curve, robot, action):
        curve_feature = self.feature_net(curve)
        x = torch.cat([curve_feature, robot, action], 1)

        q1 = self.relu(self.q1_l1(x))
        q1 = self.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = self.relu(self.q2_l1(x))
        q2 = self.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

    def Q1(self, curve, robot, action):
        curve_feature = self.feature_net(curve)
        x = torch.cat([curve_feature, robot, action], 1)

        q1 = self.relu(self.q1_l1(x))
        q1 = self.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        return q1


class TD3Agent(object):
    def __init__(self, args):
        self.max_action = args.max_action
        self.curve_dim = args.curve_normalize_dim * 3 * 2
        self.curve_feature_dim = args.curve_feature_dim
        self.robot_feature_dim = args.state_robot_dim
        self.action_dim = args.action_dim

        self.gamma = 0.99
        self.tau = 0.05
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.batch_size = 256
        # self.action_noise = OrnsteinUhlenbeckActionNoise(action_dim=self.action_dim)
        self.action_noise = GaussianActionNoise(action_dim=self.action_dim)
        self.param_noise = AdaptiveParmNoise()

        self.total_itr = 0
        self.policy_update_freq = 2

        # self.feature_net = Feature(input_dim=self.curve_dim, output_dim=self.curve_feature_dim)
        self.actor_feature_net = Feature(input_dim=self.curve_dim, output_dim=self.curve_feature_dim)
        self.critic_feature_net = Feature(input_dim=self.curve_dim, output_dim=self.curve_feature_dim)

        self.actor = Actor(feature_net=self.actor_feature_net, input_dim=self.robot_feature_dim,
                           action_dim=self.action_dim, max_action=self.max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, start_factor=1., end_factor=0.01,
                                                                 total_iters=args.max_train_episode)
        self.noise_actor = self.param_noise.get_noisy_model(self.actor)

        self.critic = Critic(feature_net=self.critic_feature_net, input_dim=self.robot_feature_dim, action_dim=self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1., end_factor=0.01,
                                                                  total_iters=args.max_train_episode)
        self.lr_decay = False

    def adapt_param_noise(self, state_curve, state_robot):
        with torch.no_grad():
            distance = torch.sqrt(torch.mean(torch.pow(self.actor(state_curve, state_robot) - self.noise_actor(state_curve, state_robot), 2)))
            self.param_noise.adapt(distance.cpu().numpy())
            self.noise_actor = self.param_noise.get_noisy_model(self.actor)

    def select_action(self, state_curve, state_robot, use_noise=True, return_q=False):
        self.actor.eval()
        with torch.no_grad():
            state_curve_tensor = torch.FloatTensor(state_curve)
            state_robot_tensor = torch.FloatTensor(state_robot)
            action_ = self.actor(state_curve_tensor, state_robot_tensor).cpu().numpy()
            if use_noise:
                action = np.clip(self.action_noise() + action_, -self.max_action, self.max_action)
                # noisy_action = self.noise_actor(state_curve_tensor, state_robot_tensor).cpu().numpy()
                # action = noisy_action
            else:
                action = action_

            if return_q:
                q = self.q_value(state_curve, state_robot, action_)
                return action, q
            else:
                return action

    def q_value(self, state_curve, state_robot, action):
        self.critic.eval()
        with torch.no_grad():
            state_curve_tensor = torch.FloatTensor(state_curve)
            state_robot_tensor = torch.FloatTensor(state_robot)
            action_tensor = torch.FloatTensor(action)
            q = self.critic.Q1(state_curve_tensor, state_robot_tensor, action_tensor)
            return q.cpu().numpy()

    def train(self, replayer_buffer, gradient_steps=1):
        self.actor.train()
        self.critic.train()

        if replayer_buffer.size < self.batch_size:
            return

        for _ in range(gradient_steps):
            self.total_itr += 1

            state_curve, state_robot, action, reward, next_state_curve, next_state_robot, done, success = replayer_buffer.sample(self.batch_size)

            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state_curve, next_state_robot) + noise).clamp(-self.max_action, self.max_action)
                target_q1, target_q2 = self.critic_target(next_state_curve, next_state_robot, next_action)
                q_next = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * (1 - success) * self.gamma * q_next

            q1, q2 = self.critic(state_curve, state_robot, action)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.total_itr % self.policy_update_freq == 0:
                on_policy_actions = self.actor(state_curve, state_robot)
                on_policy_q1 = self.critic.Q1(state_curve, state_robot, on_policy_actions)
                actor_loss = -on_policy_q1.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # state_curve, state_robot, action, reward, next_state_curve, next_state_robot, done, success = replayer_buffer.sample(self.batch_size)
        # self.adapt_param_noise(state_curve, state_robot)

        if self.lr_decay:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

    def save(self, path):
        # torch.save(self.feature_net.state_dict(), path + os.sep + "feature_net.pth")
        torch.save(self.actor_feature_net.state_dict(), path + os.sep + "actor_feature_net.pth")
        torch.save(self.critic_feature_net.state_dict(), path + os.sep + "critic_feature_net.pth")
        torch.save(self.actor.state_dict(), path + os.sep + "actor.pth")
        torch.save(self.actor_target.state_dict(), path + os.sep + "actor_target.pth")
        torch.save(self.critic.state_dict(), path + os.sep + "critic.pth")
        torch.save(self.critic_target.state_dict(), path + os.sep + "critic_target.pth")

    def load(self, path):
        # self.feature_net.load_state_dict(torch.load(path + os.sep + "feature_net.pth"))
        self.actor_feature_net.load_state_dict(torch.load(path + os.sep + "actor_feature_net.pth"))
        self.critic_feature_net.load_state_dict(torch.load(path + os.sep + "critic_feature_net.pth"))
        self.actor.load_state_dict(torch.load(path + os.sep + "actor.pth"))
        self.actor_target.load_state_dict(torch.load(path + os.sep + "actor_target.pth"))
        self.critic.load_state_dict(torch.load(path + os.sep + "critic.pth"))
        self.critic_target.load_state_dict(torch.load(path + os.sep + "critic_target.pth"))
