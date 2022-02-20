import sys

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import time

from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim

from trajectory_utils import Primitive, BreakPoint, read_base_data, read_js_data
from curve_normalization import PCA_normalization, fft_feature

sys.path.append('../greedy_fitting/')
from greedy_simplified import greedy_fit

sys.path.append('../toolbox')
from robots_def import *

Curve_Error = namedtuple("Curve_Error", ('primitive', 'max_error'))
State = namedtuple("State", ("longest_type", "curve_features", "lengths"))
Action = namedtuple("Action", ("Type", "Length"))
Memory = namedtuple('Memory', ('state', 'action', 'reward', 'next_state', 'next_action'))

ERROR_THRESHOLD = 1.0
MAX_GREEDY_STEP = 10

EPS_START = 0.5
EPS_END = 0.01
LEARNING_RATE = 0.01
GAMMA = 0.99
BATCH_SIZE = 16

REWARD_FINISH = 2000  # -------------------------------------- HERE
REWARD_DECAY_FACTOR = 0.9
REWARD_STEP = -10


def greedy_data_to_dict(greedy_data: pd.DataFrame):
    ret_dict = dict()
    for index, row in greedy_data.iterrows():
        ret_dict[row['id']] = row['n_primitives']
    return ret_dict


# TODO: Change to the latest greedy with backproj
def greedy_fit_primitive(last_bp, curve, p):
    longest_fits = {'C': None, 'L': None}

    search_left_c = search_left_l = last_bp
    search_right_c = search_right_l = len(curve)
    search_point_c = search_right_c
    search_point_l = search_right_l

    step_count = 0
    while step_count < MAX_GREEDY_STEP:
        step_count += 1

        # moveC
        left_bp = BreakPoint(idx=last_bp)
        right_bp = BreakPoint(idx=search_point_c)
        target_curve = curve[last_bp:max(last_bp+1, int(search_point_c) + 1)]
        # print(len(target_curve), last_bp, search_point_c)

        if len(target_curve) >= 2:
            primitive_c = Primitive(start=left_bp, end=right_bp, move_type='C')
            # print("---", len(target_curve), last_bp, search_point_c, step_count)
            curve_fit, max_error = primitive_c.curve_fit(target_curve, p=p)
            if max_error <= ERROR_THRESHOLD:
                if longest_fits['C'] is None or primitive_c > longest_fits['C'].primitive:
                    longest_fits['C'] = Curve_Error(primitive_c, max_error)

                search_left_c = search_point_c
                search_point_c = np.floor(np.mean([search_point_c, search_right_c]))
            else:
                search_right_c = search_point_c
                search_point_c = np.floor(np.mean([search_left_c, search_point_c]))

        # moveL
        left_bp = BreakPoint(idx=last_bp)
        right_bp = BreakPoint(idx=search_point_l)
        target_curve = curve[last_bp:max(last_bp+1, int(search_point_l) + 1)]

        primitive_l = Primitive(start=left_bp, end=right_bp, move_type='L')
        curve_fit, max_error = primitive_l.curve_fit(target_curve)
        if max_error <= ERROR_THRESHOLD:
            if longest_fits['L'] is None or primitive_l > longest_fits['L'].primitive:
                longest_fits['L'] = Curve_Error(primitive_l, max_error)

            search_left_l = search_point_l
            search_point_l = np.floor(np.mean([search_point_l, search_right_l]))
        else:
            search_right_l = search_point_l
            search_point_l = np.floor(np.mean([search_left_l, search_point_l]))

    longest_type = 'L'
    if longest_fits['L'] is None or (longest_fits['C'] is not None and
                                     longest_fits['C'].primitive > longest_fits['L'].primitive):
        longest_type = 'C'
    elif longest_fits['C'] is None or (longest_fits['L'] is not None and
                                       longest_fits['L'].primitive >= longest_fits['C'].primitive):
        longest_type = 'L'

    return longest_fits, longest_type


def reward_function(i_step, done):
    reward = REWARD_STEP
    if done:
        reward += REWARD_FINISH * (REWARD_DECAY_FACTOR ** i_step)

    return reward


class ReplayMemory(object):
    def __init__(self, capacity=1000):
        self.memory = deque([], maxlen=capacity)

    def push(self, memory):
        self.memory.append(memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.input = nn.Linear(input_dim, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_dim)

        # self.drop_out = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden1(x))
        x = self.output(x)
        return x


class RL_Agent(object):

    def __init__(self, n_curve_feature: int, n_length: int, lr: float = LEARNING_RATE):
        self.input_dim = n_curve_feature * 3 + 1
        self.output_dim = 1
        self._n_curve_feature = n_curve_feature
        self._n_length = n_length
        self.lr = lr
        self.gamma = GAMMA
        self.memory = ReplayMemory()
        self.batch_size = BATCH_SIZE

        self.LL_policy_net = DQN(self.input_dim, self.output_dim)
        self.LC_policy_net = DQN(self.input_dim, self.output_dim)
        self.CL_policy_net = DQN(self.input_dim, self.output_dim)
        self.CC_policy_net = DQN(self.input_dim, self.output_dim)
        self.policy_DQNs = {'LL': self.LL_policy_net, 'LC': self.LC_policy_net,
                            'CL': self.CL_policy_net, 'CC': self.CC_policy_net}

        self.LL_target_net = DQN(self.input_dim, self.output_dim)
        self.LL_target_net.load_state_dict(self.LL_policy_net.state_dict())
        self.LC_target_net = DQN(self.input_dim, self.output_dim)
        self.LC_target_net.load_state_dict(self.LC_policy_net.state_dict())
        self.CL_target_net = DQN(self.input_dim, self.output_dim)
        self.CL_target_net.load_state_dict(self.CL_policy_net.state_dict())
        self.CC_target_net = DQN(self.input_dim, self.output_dim)
        self.CC_target_net.load_state_dict(self.CC_policy_net.state_dict())
        self.target_DQNs = {'LL': self.LL_target_net, 'LC': self.LC_target_net,
                            'CL': self.CL_target_net, 'CC': self.CC_target_net}

        self.optimizer_LL = optim.RMSprop(self.LL_policy_net.parameters(), lr=self.lr)
        self.optimizer_LC = optim.RMSprop(self.LC_policy_net.parameters(), lr=self.lr)
        self.optimizer_CL = optim.RMSprop(self.CL_policy_net.parameters(), lr=self.lr)
        self.optimizer_CC = optim.RMSprop(self.CC_policy_net.parameters(), lr=self.lr)
        self.optimizers = {'LL': self.optimizer_LL, 'LC': self.optimizer_LC,
                           'CL': self.optimizer_CL, 'CC': self.optimizer_CC}

        self.criterion = nn.SmoothL1Loss()

    def get_action(self, state: State, epsilon: float = 0., valid_types: dict = None):
        n_action = len(state.lengths)
        actions_tensor = torch.reshape(torch.tensor(state.lengths), (n_action, 1))
        curve_feature_tensor = torch.tensor(state.curve_features)
        curve_feature_tensor = curve_feature_tensor.repeat(n_action, 1)
        x_tensor = torch.hstack((curve_feature_tensor, actions_tensor))

        action = Action(Type=None, Length=None)
        q_value = 0
        nn_activated = ''
        sample = np.random.rand()

        if state.longest_type == 'L':
            with torch.no_grad():
                q_ll = self.LL_policy_net(x_tensor.float())
                q_lc = self.LC_policy_net(x_tensor.float())
            length_ll = (torch.argmax(q_ll) + 1) * 1/n_action
            length_lc = (torch.argmax(q_lc) + 1) * 1/n_action
            primitive_type = 'L' if torch.max(q_ll) >= torch.max(q_lc) or not valid_types['C'] else 'C'
            primitive_length = length_ll if primitive_type == 'L' else length_lc
            q_value = torch.max(q_ll) if primitive_type == 'L' else torch.max(q_lc)

            if sample < epsilon:
                primitive_type = 'C' if primitive_type == 'L' and valid_types['C'] else 'L'
                q_value_idx = np.random.randint(1, n_action+1)
                q_value = q_ll[q_value_idx-1] if primitive_type == 'L' else q_lc[q_value_idx-1]
                primitive_length = q_value_idx * 1/n_action

            action = Action(Type=primitive_type, Length=primitive_length)
            nn_activated = 'L{}'.format(primitive_type)

        elif state.longest_type == 'C':
            with torch.no_grad():
                q_cl = self.CL_policy_net(x_tensor.float())
                q_cc = self.CC_policy_net(x_tensor.float())
            length_ll = (torch.argmax(q_cl) + 1) * 1/n_action
            length_lc = (torch.argmax(q_cc) + 1) * 1/n_action
            primitive_type = 'L' if torch.max(q_cl) >= torch.max(q_cc) or not valid_types['C'] else 'C'
            primitive_length = length_ll if primitive_type == 'L' else length_lc
            q_value = torch.max(q_cl) if primitive_type == 'L' else torch.max(q_cc)

            if sample < epsilon:
                primitive_type = 'C' if primitive_type == 'L' and valid_types['C'] else 'L'
                q_value_idx = np.random.randint(1, n_action+1)
                q_value = q_cl[q_value_idx - 1] if primitive_type == 'L' else q_cc[q_value_idx - 1]
                primitive_length = q_value_idx * 1/n_action

            action = Action(Type=primitive_type, Length=primitive_length)
            nn_activated = 'C{}'.format(primitive_type)
        q_value = torch.reshape(q_value, (1, 1))
        return action, q_value, nn_activated

    def memory_save(self, state, action, reward, next_state, next_action):
        memory = Memory(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action)
        self.memory.push(memory)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        memory_batch = self.memory.sample(self.batch_size)

        for memory in memory_batch:
            state = memory.state
            action = memory.action
            reward = memory.reward
            next_state = memory.next_state
            next_action = memory.next_action

            nn_activated = '{}{}'.format(state.longest_type, action.Type)
            n_action = len(state.lengths)
            actions_tensor = torch.reshape(torch.tensor(state.lengths), (n_action, 1))
            curve_feature_tensor = torch.tensor(state.curve_features)
            curve_feature_tensor = curve_feature_tensor.repeat(n_action, 1)
            x_tensor = torch.hstack((curve_feature_tensor, actions_tensor))
            q_values = self.policy_DQNs[nn_activated](x_tensor.float())
            q_value = q_values[int(np.floor(action.Length * n_action)) - 1]
            q_value = q_value.reshape(1)

            q_next = 0.
            if next_state is not None:
                next_nn_activated = '{}{}'.format(next_state.longest_type, next_action.Type)
                n_action = len(next_state.lengths)
                next_actions_tensor = torch.reshape(torch.tensor(next_state.lengths), (n_action, 1))
                next_curve_feature_tensor = torch.tensor(next_state.curve_features)
                next_curve_feature_tensor = next_curve_feature_tensor.repeat(n_action, 1)
                next_x_tensor = torch.hstack((next_curve_feature_tensor, next_actions_tensor))
                q_next = self.target_DQNs[next_nn_activated](next_x_tensor.float())
                q_next = q_next[int(np.floor(next_action.Length * n_action)) - 1]
                q_next = q_next.reshape(1)

            expected_q = q_next * self.gamma + reward
            if type(expected_q) is float:
                expected_q = torch.tensor(expected_q).reshape(1)
            optimizer = self.optimizers[nn_activated]
            optimizer.zero_grad()
            loss = self.criterion(q_value, expected_q)
            loss.backward()
            optimizer.step()

    def update_target_nets(self):
        self.LL_target_net.load_state_dict(self.LL_policy_net.state_dict())
        self.LC_target_net.load_state_dict(self.LC_policy_net.state_dict())
        self.CL_target_net.load_state_dict(self.CL_policy_net.state_dict())
        self.CC_target_net.load_state_dict(self.CC_policy_net.state_dict())


class RL_Env(object):
    def __init__(self, target_curve, target_curve_normal, target_curve_js, greedy_obj, n_feature=10, n_action=10):
        self.target_curve = target_curve
        self.target_curve_normal = target_curve_normal
        self.target_curve_js = target_curve_js
        self.greedy_obj = greedy_obj
        self.fit_curve = [target_curve[0]]
        self.last_bp = 0

        self.primitives = []
        self.n_feature = n_feature
        self.n_action = n_action
        self.lengths = [(x + 1) * 0.1 for x in range(self.n_action)]

        self.done = False
        self.i_step = 0

        self.longest_primitives = None

    def reset(self, target_curve=None):
        if target_curve is not None:
            self.target_curve = target_curve
        self.fit_curve = [self.target_curve[0]]
        self.last_bp = 0

        self.primitives = []

        self.done = False
        self.i_step = 0

        remaining_curve = self.target_curve[self.last_bp:, :]
        normalized_curve = PCA_normalization(remaining_curve)
        curve_features, _ = fft_feature(normalized_curve, self.n_feature)

        self.longest_primitives, longest_type = greedy_fit_primitive(last_bp=self.last_bp, curve=self.target_curve,
                                                                     p=self.fit_curve[-1])
        valid_types = {"L": self.longest_primitives['L'] is not None,
                       "C": self.longest_primitives['C'] is not None}
        state = State(longest_type=longest_type, curve_features=curve_features, lengths=self.lengths)
        done = False
        return state, done, valid_types

    def step(self, action: Action, i_step: int):
        self.i_step = i_step
        primitive_type = action.Type
        primitive_length = action.Length
        next_primitive = self.longest_primitives[primitive_type].primitive.curve
        primitive_end_index = int(np.ceil(len(next_primitive) * primitive_length))
        new_curve = next_primitive[0:primitive_end_index]

        self.fit_curve = np.concatenate([self.fit_curve, new_curve[1:, :]], axis=0)
        self.last_bp = len(self.fit_curve)-1
        done = len(self.fit_curve) >= len(self.target_curve)
        reward = reward_function(i_step, done)
        if done:
            return None, reward, done, None

        self.longest_primitives, longest_type = greedy_fit_primitive(last_bp=self.last_bp, curve=self.target_curve,
                                                                     p=self.fit_curve[-1])
        valid_types = {"L": self.longest_primitives['L'] is not None,
                       "C": self.longest_primitives['C'] is not None}
        remaining_curve = self.target_curve[self.last_bp:, :]
        normalized_curve = PCA_normalization(remaining_curve)
        curve_features, _ = fft_feature(normalized_curve, self.n_feature)
        state = State(longest_type=longest_type, curve_features=curve_features, lengths=self.lengths)
        # done = len(self.fit_curve) >= len(self.target_curve)
        # reward = reward_function(i_step, done)

        return state, reward, done, valid_types

    def update_greedy_obj(self):
        pass


def train_rl(agent: RL_Agent, curve_base_data, curve_js_data, n_episode=1000):
    robot = abb6640()
    print("Train Start")

    for i_episode in range(n_episode):
        timer = time.time_ns()

        epsilon = EPS_START - i_episode * (EPS_START - EPS_END) / n_episode

        episode_reward = 0

        random_curve_idx = np.random.randint(0, len(curve_base_data))
        curve_base, curve_normal = curve_base_data[random_curve_idx]
        curve_js = curve_js_data[random_curve_idx]

        greedy_fit_obj = greedy_fit(robot, curve_base, curve_normal, curve_js, d=50)
        greedy_fit_obj.primitives = {'movel_fit':greedy_fit_obj.movel_fit_greedy,
                                     'movec_fit':greedy_fit_obj.movec_fit_greedy}

        env = RL_Env(target_curve=curve_base, target_curve_normal=curve_normal, target_curve_js=curve_js,
                     greedy_obj=greedy_fit_obj)

        print("Episode Start")

        state, done, valid_actions = env.reset()
        action, q_value, nn_activated = agent.get_action(state, epsilon=epsilon, valid_types=valid_actions)

        i_step = 0
        while not done:
            i_step += 1
            print("{}/{}".format(len(env.fit_curve), len(env.target_curve)))
            print("Action:", action, valid_actions)

            next_state, reward, done, valid_actions = env.step(action, i_step)
            next_action = None
            if not done:
                next_action, next_q_value, next_nn_activated = agent.get_action(next_state, epsilon=epsilon,
                                                                                valid_types=valid_actions)

            agent.memory_save(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action)
            agent.learn()
            episode_reward += reward

            state = next_state
            action = next_action

            print("Step: {} END {}".format(i_step, reward))

        print("Episode {} / {} --- {} Steps --- Reward: {:.3f} --- {:.3f}s Curve {}"
              .format(i_episode + 1, n_episode, i_step, episode_reward,
                      (time.time_ns() - timer) / (10 ** 9), random_curve_idx))
        agent.update_target_nets()
        break


def rl_fit(curve_base_data, curve_js_data):
    n_feature = 10
    n_action = 10
    agent = RL_Agent(n_feature, n_action)
    train_rl(agent, curve_base_data, curve_js_data)


def main():
    curve_base_data = read_base_data("data/base")
    curve_js_data = read_js_data("data/js")
    rl_fit(curve_base_data, curve_js_data)


if __name__ == '__main__':
    main()

