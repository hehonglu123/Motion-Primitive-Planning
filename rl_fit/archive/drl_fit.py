import sys

import numpy as np
import pandas as pd
import random
import os
import time

from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_fit.archive.trajectory_utils import Primitive, BreakPoint, read_base_data, read_js_data
from rl_fit.archive.rl_fit import PCA_normalization, fft_feature

sys.path.append('../../greedy_fitting/')
from greedy_simplified import greedy_fit

sys.path.append('../../toolbox')
from robots_def import *

Curve_Error = namedtuple("Curve_Error", ('primitive', 'max_error'))
State = namedtuple("State", ("longest_type", "curve_features"))
Action = namedtuple("Action", ("Type", "Length"))
Memory = namedtuple('Memory', ('state', 'action', 'reward', 'next_state', 'next_action'))
primitive_type_code = {"L": 0, "C": 1, "J": 2}

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

ERROR_THRESHOLD = 1.0
MAX_GREEDY_STEP = 10

TAU_START = 10
TAU_END = 0.01
MAX_TAU_EPISODE = 0.6
LEARNING_RATE = 0.0001
GAMMA = 0.99
BATCH_SIZE = 16

REWARD_FINISH = 2000  # -------------------------------------- HERE
REWARD_DECAY_FACTOR = 0.9
REWARD_STEP = -10

SAVE_MODEL = 10


def episode_tau(i_episode, n_episode):
    # factor = min(1, i_episode/(n_episode * MAX_TAU_EPISODE))
    factor = min(1, i_episode / 200)
    tau = TAU_START - factor * (TAU_START - TAU_END)
    return tau


def greedy_data_to_dict(greedy_data: pd.DataFrame):
    ret_dict = dict()
    for index, row in greedy_data.iterrows():
        ret_dict[row['id']] = row['n_primitives']
    return ret_dict


# TODO: Change to the latest greedy with backproj_js
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
        state = memory.state
        type_encode = primitive_type_code[state.longest_type]
        state_encode = np.hstack([state.curve_features, type_encode])
        state_tensor = torch.tensor(state_encode)

        next_state = memory.next_state
        next_state_tensor = None
        if next_state is not None:
            next_type_encode = primitive_type_code[next_state.longest_type]
            next_state_encode = np.hstack([next_state.curve_features, next_type_encode])
            next_state_tensor = torch.tensor(next_state_encode)

        new_memory = Memory(state=state_tensor, action=memory.action, reward=memory.reward,
                            next_state=next_state_tensor, next_action=memory.next_action)

        self.memory.append(new_memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.input = nn.Linear(input_dim, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.hidden4 = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)

        self.drop_out = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.output(x)
        return x


class RL_Agent(object):

    def __init__(self, n_curve_feature: int, n_length: int, lr: float = LEARNING_RATE):
        self.input_dim = n_curve_feature * 3 * 2 + 1
        self.output_dim = 3
        self.n_curve_feature = n_curve_feature
        self._n_length = n_length
        self.lr = lr
        self.gamma = GAMMA
        self.memory = ReplayMemory()
        self.batch_size = BATCH_SIZE

        self.target_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim)
        self.policy_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)

        self.criterion = nn.SmoothL1Loss()

    def get_action(self, state: State, valid_types: dict = None, tau=0.01):
        with torch.no_grad():
            type_code = primitive_type_code[state.longest_type]
            curve_feature_tensor = torch.tensor(state.curve_features)
            x_tensor = torch.hstack((curve_feature_tensor, torch.tensor(type_code)))

            output = self.policy_net(x_tensor.float())
            length = torch.sigmoid(output[0])
            # primitive_probs = torch.softmax(output[1:], 0)
            primitive_probs = F.gumbel_softmax(output[1:], dim=0, tau=tau)
            primitive_type_encode = np.random.choice(2, p=primitive_probs.detach().numpy())
            primitive_type = 'C' if primitive_type_encode == 1 and valid_types['C'] else 'L'

            action = Action(Type=primitive_type, Length=length.detach().numpy())

            return action

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

            x_tensor = state
            q_value = self.policy_net(x_tensor.float())

            q_next = torch.tensor([0., 0., 0.])
            if next_state is not None:
                next_x_tensor = next_state
                q_next = self.target_net(next_x_tensor.float())

            expected_q = q_next * self.gamma + reward
            if type(expected_q) is float:
                expected_q = torch.tensor(expected_q).reshape(1)
            self.optimizer.zero_grad()
            loss = -self.criterion(q_value, expected_q)
            loss.backward()
            self.optimizer.step()

    def update_target_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + os.sep + 'DQN_policy_net.pth')


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
        state = State(longest_type=longest_type, curve_features=curve_features)
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
        state = State(longest_type=longest_type, curve_features=curve_features)
        # done = len(self.fit_curve) >= len(self.target_curve)
        # reward = reward_function(i_step, done)

        return state, reward, done, valid_types

    def update_greedy_obj(self):
        pass


def train_rl(agent: RL_Agent, curve_base_data, curve_js_data, n_episode=10000):
    robot = abb6640()
    print("Train Start")

    episode_rewards = []
    episode_steps = []
    episode_target_curve = []

    for i_episode in range(n_episode):
        timer = time.time_ns()

        tau = episode_tau(i_episode, n_episode)

        episode_reward = 0

        random_curve_idx = np.random.randint(0, len(curve_base_data))
        curve_base, curve_normal = curve_base_data[random_curve_idx]
        curve_js = curve_js_data[random_curve_idx]

        greedy_fit_obj = greedy_fit(robot, curve_base, curve_normal, curve_js, d=50)
        greedy_fit_obj.primitives = {'movel_fit':greedy_fit_obj.movel_fit_greedy,
                                     'movec_fit':greedy_fit_obj.movec_fit_greedy}

        env = RL_Env(target_curve=curve_base, target_curve_normal=curve_normal, target_curve_js=curve_js,
                     greedy_obj=greedy_fit_obj, n_feature=agent.n_curve_feature)

        # print("Episode Start")

        state, done, valid_actions = env.reset()
        action = agent.get_action(state, valid_types=valid_actions, tau=tau)

        i_step = 0
        while not done:
            i_step += 1
            # print("{}/{}".format(len(env.fit_curve), len(env.target_curve)))
            # print("Action:", action, valid_actions)

            next_state, reward, done, valid_actions = env.step(action, i_step)
            next_action = None
            if not done:
                next_action = agent.get_action(next_state, valid_types=valid_actions, tau=tau)

            agent.memory_save(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action)
            agent.learn()
            episode_reward += reward

            state = next_state
            action = next_action

            # print("Step: {} END {}".format(i_step, reward))

        print("Episode {} / {} --- {} Steps --- Reward: {:.3f} --- {:.3f}s Curve {}"
              .format(i_episode + 1, n_episode, i_step, episode_reward,
                      (time.time_ns() - timer) / (10 ** 9), random_curve_idx))
        agent.update_target_nets()
        episode_rewards.append(episode_reward)
        episode_steps.append(i_step)
        episode_target_curve.append(random_curve_idx)

        if (i_episode + 1) % SAVE_MODEL == 0:
            print("=== Saving Model ===")
            save_data(episode_rewards, episode_steps, episode_target_curve)
            agent.save_model('../model')
            print("DQNs saved at '{}'".format('model/'))
            print("======")

    return episode_rewards, episode_steps


def save_data(episode_rewards, episode_steps, episode_target_curve):
    df = pd.DataFrame({"episode_rewards": episode_rewards,
                       "episode_steps": episode_steps,
                       "curve": episode_target_curve})
    df.to_csv("Training Curve Data.csv")


def rl_fit(curve_base_data, curve_js_data):
    n_feature = 20
    n_action = 10
    agent = RL_Agent(n_feature, n_action)
    train_rl(agent, curve_base_data, curve_js_data)


def main():
    curve_base_data = read_base_data("../train_data/base")
    curve_js_data = read_js_data("../train_data/js")
    rl_fit(curve_base_data, curve_js_data)


if __name__ == '__main__':
    main()

