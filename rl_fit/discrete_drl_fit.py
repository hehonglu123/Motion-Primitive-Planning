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
import torch.nn.functional as F
import torch.optim as optim

from trajectory_utils import Primitive, BreakPoint, read_base_data, read_js_data
from curve_normalization import PCA_normalization, fft_feature
from cnn_encoder import load_encoder

sys.path.append('../greedy_fitting/')
from greedy_simplified import greedy_fit

sys.path.append('../toolbox')
from robots_def import *

Curve_Error = namedtuple("Curve_Error", ('primitive', 'max_error'))
State = namedtuple("State", ("longest_type", "curve_features"))
Action = namedtuple("Action", ("Type", "Length", "Code"))
Memory = namedtuple('Memory', ('state', 'action', 'reward', 'next_state', 'done'))
primitive_type_code = {"L": 0, "C": 1, "J": 2}

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

ERROR_THRESHOLD = 1.0
MAX_GREEDY_STEP = 10

EPS_START = 0.5
EPS_END = 0.01
EPS_DECAY = 10

TAU_START = 10
TAU_END = 0.01
MAX_TAU_EPISODE = 0.6
LEARNING_RATE = 0.001
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

        action = memory.action
        action_type_encode = action.Code

        next_state = memory.next_state
        next_state_encode = np.zeros_like(state_encode)
        if next_state is not None:
            next_type_encode = primitive_type_code[next_state.longest_type]
            next_state_encode = np.hstack([next_state.curve_features, next_type_encode])

        new_memory = Memory(state=state_encode, action=action_type_encode, reward=memory.reward,
                            next_state=next_state_encode, done=memory.done)

        self.memory.append(new_memory)

    def sample(self, batch_size):
        sample_batch = random.sample(self.memory, batch_size)
        state_batch = np.vstack([m.state for m in sample_batch])
        action_batch = np.array([m.action for m in sample_batch])
        reward_batch = np.array([m.reward for m in sample_batch])
        next_state_batch = np.vstack([m.next_state for m in sample_batch])
        done_batch = np.array([m.done for m in sample_batch])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        self.input = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 256)
        # self.hidden3 = nn.Linear(128, 128)
        # self.hidden4 = nn.Linear(128, 128)
        self.output = nn.Linear(256, output_dim)

        self.drop_out = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        # x = self.relu(self.hidden3(x))
        # x = self.relu(self.hidden4(x))
        x = self.output(x)
        return x


class RL_Agent(object):

    def __init__(self, n_curve_feature: int, n_action: int, lr: float = LEARNING_RATE):
        self.input_dim = n_curve_feature + 1
        # self.input_dim = n_curve_feature * 3 * 2 + 1
        self.output_dim = n_action * 2
        self.n_curve_feature = n_curve_feature
        self.n_action = n_action
        self.lr = lr
        self.gamma = GAMMA
        self.memory = ReplayMemory()
        self.batch_size = BATCH_SIZE

        self.target_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim)
        self.policy_net = DQN(input_dim=self.input_dim, output_dim=self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)

        self.criterion = nn.SmoothL1Loss()

    def get_action(self, state: State, valid_types: dict = None, tau=0.01, epsilon=0.):
        self.policy_net.eval()
        with torch.no_grad():
            type_code = primitive_type_code[state.longest_type]
            curve_feature_tensor = torch.tensor(state.curve_features)
            x_tensor = torch.hstack((curve_feature_tensor, torch.tensor(type_code)))

            # action_softmax = torch.softmax(output, dim=0)

            # action_code = torch.argmax(action_softmax)
            # action_softmax = F.gumbel_softmax(output, tau=tau, dim=0)
            # action_code = np.random.choice(self.output_dim, p=action_softmax.detach().numpy())

            eps_sample = np.random.rand()
            if eps_sample < epsilon:
                action_code = np.random.randint(0, self.output_dim)
            else:
                output = self.policy_net(x_tensor.float())
                action_code = torch.argmax(output).data.detach().numpy()

            primitive_type = 'C' if action_code >= self.n_action and valid_types['C'] else 'L'

            length = ((action_code % self.n_action) + 1) * (1 / self.n_action)

            action = Action(Type=primitive_type, Length=length, Code=action_code)

            return action

    def memory_save(self, state, action, reward, next_state, done):
        memory = Memory(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.push(memory)

    def td_estimate(self, state, action):
        state_tensor = torch.from_numpy(state).float()
        q_value = self.policy_net(state_tensor)
        state_action_value = q_value[:, action]
        return state_action_value

    def td_target(self, reward, next_state, done):

        with torch.no_grad():
            next_state_tensor = torch.from_numpy(next_state).float()
            reward_tensor = torch.from_numpy(reward).float()
            done_tensor = torch.from_numpy(done).float()
            next_q_value = self.policy_net(next_state_tensor)
            best_action = torch.argmax(next_q_value, dim=1)
            next_q_target = self.target_net(next_state_tensor)
            next_state_action_value = next_q_target[:, best_action]
            expected_reward = reward_tensor + (1 - done_tensor) * self.gamma * next_state_action_value
            return expected_reward

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        self.policy_net.train()

        td_est = self.td_estimate(state_batch, action_batch)
        td_tgt = self.td_target(reward_batch, next_state_batch, done_batch)

        self.optimizer.zero_grad()
        loss = self.criterion(td_est, td_tgt)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def update_target_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + os.sep + 'DQN_policy_net.pth')

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path), path + os.sep + 'DQN_policy_net.pth')
        self.target_net.load_state_dict(self.policy_net.state_dict())


class RL_Env(object):
    def __init__(self, target_curve, target_curve_normal, target_curve_js, greedy_obj, feature_encoder,
                 n_feature=10, n_action=10):
        self.target_curve = target_curve
        self.target_curve_normal = target_curve_normal
        self.target_curve_js = target_curve_js
        self.greedy_obj = greedy_obj
        self.fit_curve = [target_curve[0]]
        self.last_bp = 0

        self.encoder = feature_encoder

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
        # curve_features, _ = fft_feature(normalized_curve, self.n_feature)
        normalized_curve_tensor = torch.from_numpy(np.array([normalized_curve.T])).float()
        curve_features = self.encoder(normalized_curve_tensor)
        curve_features = curve_features.detach().numpy().flatten()

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
        # curve_features, _ = fft_feature(normalized_curve, self.n_feature)
        normalized_curve_tensor = torch.from_numpy(np.array([normalized_curve.T])).float()
        curve_features = self.encoder(normalized_curve_tensor)
        curve_features = curve_features.detach().numpy().flatten()

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

    feature_encoder = load_encoder('cnn_model/cnn_model.pth')
    feature_encoder.eval()

    epsilon = EPS_START

    for i_episode in range(n_episode):
        timer = time.time_ns()

        tau = episode_tau(i_episode, n_episode)
        # epsilon = EPS_START - i_episode * (EPS_START - EPS_END) / n_episode
        if (i_episode + 1) % EPS_DECAY == 0:
            epsilon = epsilon * 0.995

        episode_reward = 0

        random_curve_idx = np.random.randint(0, len(curve_base_data))
        curve_base, curve_normal = curve_base_data[random_curve_idx]
        curve_js = curve_js_data[random_curve_idx]

        greedy_fit_obj = greedy_fit(robot, curve_base, curve_normal, curve_js, d=50)
        greedy_fit_obj.primitives = {'movel_fit': greedy_fit_obj.movel_fit_greedy,
                                     'movec_fit': greedy_fit_obj.movec_fit_greedy}

        env = RL_Env(target_curve=curve_base, target_curve_normal=curve_normal, target_curve_js=curve_js,
                     greedy_obj=greedy_fit_obj, n_feature=agent.n_curve_feature, feature_encoder=feature_encoder)

        # print("Episode Start")

        state, done, valid_actions = env.reset()

        i_step = 0
        while not done:
            i_step += 1
            # print("{}/{}".format(len(env.fit_curve), len(env.target_curve)))
            # print("Action:", action, valid_actions)

            action = agent.get_action(state, valid_types=valid_actions, tau=tau, epsilon=epsilon)
            # print(action)

            next_state, reward, done, valid_actions = env.step(action, i_step)

            agent.memory_save(state=state, action=action, reward=reward, next_state=next_state, done=done)
            agent.learn()
            episode_reward += reward

            state = next_state

            if done:
                break

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
            agent.save_model('model')
            print("DQNs saved at '{}'".format('model/'))
            print("======")

    return episode_rewards, episode_steps


def evaluate_rl(agent: RL_Agent, curve_base_data, curve_js_data):
    robot = abb6640()
    print("Evaluation Start")

    episode_steps = []
    episode_target_curve = []

    feature_encoder = load_encoder('cnn_model/cnn_model.pth')
    feature_encoder.eval()

    for i, (curve_base, curve_normal) in enumerate(curve_base_data):
        episode_reward = 0
        curve_js = curve_js_data[i]
        timer = time.time_ns()

        greedy_fit_obj = greedy_fit(robot, curve_base, curve_normal, curve_js, d=50)
        greedy_fit_obj.primitives = {'movel_fit': greedy_fit_obj.movel_fit_greedy,
                                     'movec_fit': greedy_fit_obj.movec_fit_greedy}

        env = RL_Env(target_curve=curve_base, target_curve_normal=curve_normal, target_curve_js=curve_js,
                     greedy_obj=greedy_fit_obj, n_feature=agent.n_curve_feature, feature_encoder=feature_encoder)

        # print("Episode Start")

        state, done, valid_actions = env.reset()

        i_step = 0
        while not done:
            i_step += 1
            # print("{}/{}".format(len(env.fit_curve), len(env.target_curve)))
            # print("Action:", action, valid_actions)

            action = agent.get_action(state, valid_types=valid_actions, epsilon=0)
            # print(action)

            next_state, reward, done, valid_actions = env.step(action, i_step)

            agent.memory_save(state=state, action=action, reward=reward, next_state=next_state, done=done)
            episode_reward += reward

            state = next_state

            if done:
                break

            # print("Step: {} END {}".format(i_step, reward))

        print("Curve {} / {} --- {} Steps --- Reward: {:.3f} --- {:.3f}s "
              .format(i, len(curve_base_data), i_step, episode_reward,
                      (time.time_ns() - timer) / (10 ** 9)))

        episode_steps.append(i_step)
        episode_target_curve.append(i)


def save_data(episode_rewards, episode_steps, episode_target_curve):
    df = pd.DataFrame({"episode_rewards": episode_rewards,
                       "episode_steps": episode_steps,
                       "curve": episode_target_curve})
    df.to_csv("Training Curve Data.csv")


def rl_fit(curve_base_data, curve_js_data):
    n_feature = 128
    n_action = 10
    agent = RL_Agent(n_feature, n_action)
    train_rl(agent, curve_base_data, curve_js_data)


def main():
    curve_base_data = read_base_data("data/base")
    curve_js_data = read_js_data("data/js")
    # rl_fit(curve_base_data, curve_js_data)

    n_feature = 128
    n_action = 10
    agent = RL_Agent(n_feature, n_action)
    agent.load_model('model')
    evaluate_rl(agent, curve_base_data, curve_js_data)


if __name__ == '__main__':
    main()

