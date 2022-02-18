import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import time

from collections import namedtuple
import torch
import torch.nn as nn

from trajectory_utils import Primitive, BreakPoint, read_data
from curve_normalization import PCA_normalization, fft_feature

Curve_Error = namedtuple("Curve_Error", ('primitive', 'max_error'))
State = namedtuple("State", ("longest_type", "curve_features", "lengths"))
Action = namedtuple("Action", ("Type", "Length"))

ERROR_THRESHOLD = 1.0
MAX_GREEDY_STEP = 10

EPS_START = 0.5
EPS_END = 0.01
LEARNING_RATE = 0.01

REWARD_FINISH = 2000  # -------------------------------------- HERE
REWARD_DECAY_FACTOR = 0.9
REWARD_STEP = -10


def greedy_data_to_dict(greedy_data: pd.DataFrame):
    ret_dict = dict()
    for index, row in greedy_data.iterrows():
        ret_dict[row['id']] = row['n_primitives']
    return ret_dict


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
        target_curve = curve[last_bp:int(search_point_c) + 1]
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
        target_curve = curve[last_bp:int(search_point_l) + 1]

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

    longest_type = 'L' if longest_fits['L'].primitive >= longest_fits['C'] else 'C'
    return longest_fits, longest_type


def reward_function(i_step, done):
    reward = REWARD_STEP
    if done:
        reward += REWARD_FINISH * (REWARD_DECAY_FACTOR ** i_step)

    return reward


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

    def __init__(self, n_curve_feature: int, n_length: int):
        self.input_dim = n_curve_feature, n_length
        self.output_dim = n_length
        self._n_curve_feature = n_curve_feature
        self._n_length = n_length

        self.LL_NN = DQN(self.input_dim, self.output_dim)
        self.LC_NN = DQN(self.input_dim, self.output_dim)
        self.CL_NN = DQN(self.input_dim, self.output_dim)
        self.CC_NN = DQN(self.input_dim, self.output_dim)

    def get_action(self, state: State, epsilon: float = 0.):
        n_action = len(state.lengths)
        actions_tensor = torch.reshape(torch.tensor(state.lengths), (n_action, 1))
        curve_feature_tensor = torch.tensor(state.curve_features)
        curve_feature_tensor = curve_feature_tensor.repeat(n_action, 1)
        x_tensor = torch.hstack((curve_feature_tensor, actions_tensor))

        action = Action(Type=None, Length=None)

        sample = np.random.rand()

        if state.longest_type == 'L':
            q_ll = self.LL_NN(x_tensor)
            q_lc = self.LC_NN(x_tensor)
            length_ll = (torch.argmax(q_ll) + 1) * 0.1
            length_lc = (torch.argmax(q_lc) + 1) * 0.1
            primitive_type = 'L' if torch.max(q_ll) > torch.max(q_lc) else 'C'
            primitive_length = length_ll if primitive_type is 'L' else length_lc

            if sample < epsilon:
                primitive_type = 'C' if primitive_type == 'L' else 'L'
                primitive_length = np.random.randint(1, 11) * 0.1

            action = Action(Type=primitive_type, Length=primitive_length)

        elif state.longest_type == 'C':
            q_cl = self.CL_NN(x_tensor)
            q_cc = self.CC_NN(x_tensor)
            length_ll = (torch.argmax(q_cl) + 1) * 0.1
            length_lc = (torch.argmax(q_cc) + 1) * 0.1
            primitive_type = 'L' if torch.max(q_cl) > torch.max(q_cc) else 'C'
            primitive_length = length_ll if primitive_type is 'L' else length_lc

            if sample < epsilon:
                primitive_type = 'C' if primitive_type == 'L' else 'L'
                primitive_length = np.random.randint(1, 11) * 0.1

            action = Action(Type=primitive_type, Length=primitive_length)

        return action

    # TODO: Learning and Back-propagation
    def learn(self, q, q_next, reward):
        pass


class RL_Env(object):
    def __init__(self, target_curve):
        self.target_curve = target_curve
        self.fit_curve = [target_curve[0]]
        self.last_bp = 0

        self.primitives = []
        self.lengths = [(x + 1) * 0.1 for x in range(10)]

        self.done = False
        self.i_step = 0

        self.longest_primitives = None

    def reset(self, target_curve=None):
        if target_curve is not None:
            self.target_curve = target_curve
        self.fit_curve = [target_curve[0]]
        self.last_bp = 0

        self.primitives = []

        self.done = False
        self.i_step = 0

        remaining_curve = self.target_curve[self.last_bp, :]
        normalized_curve = PCA_normalization(remaining_curve)
        curve_features = fft_feature(normalized_curve, 10)

        self.longest_primitives, longest_type = greedy_fit_primitive(last_bp=self.last_bp, curve=self.target_curve,
                                                                     p=self.fit_curve[-1])

        state = State(longest_type=longest_type, curve_features=curve_features, lengths=self.lengths)
        done = False
        return state, done

    def step(self, action: Action, i_step: int):
        self.i_step = i_step
        primitive_type = action.Type
        primitive_length = action.Length
        next_primitive = self.longest_primitives[primitive_type].primitive.curve
        primitive_end_index = int(np.ceil(len(next_primitive) * primitive_length))
        new_curve = next_primitive[0:primitive_end_index]

        self.fit_curve = np.concatenate([self.fit_curve, new_curve], axis=0)
        self.last_bp = len(self.fit_curve)-1

        self.longest_primitives, longest_type = greedy_fit_primitive(last_bp=self.last_bp, curve=self.target_curve,
                                                                     p=self.fit_curve[-1])
        remaining_curve = self.target_curve[self.last_bp, :]
        normalized_curve = PCA_normalization(remaining_curve)
        curve_features = fft_feature(normalized_curve, 10)
        state = State(longest_type=longest_type, curve_features=curve_features, lengths=self.lengths)
        done = len(self.fit_curve) >= len(self.target_curve)
        reward = reward_function(i_step, done)

        return state, reward, done


def train_rl(agent: RL_Agent, data, n_episode=1000):
    for i_episode in range(n_episode):
        epsilon = EPS_START - i_episode * (EPS_START - EPS_END) / n_episode

        episode_reward = 0
        lr = LEARNING_RATE

        random_curve_idx = np.random.randint(0, data)
        curve = data[random_curve_idx]

        env = RL_Env(target_curve=curve)
        state, done = env.reset()
        action = agent.get_action(state, epsilon=epsilon)

        i_step = 0
        while not done:
            i_step += 1
            next_state, reward, done = env.step(action, i_step)
            next_action = agent.get_action(next_state, epsilon=epsilon)

