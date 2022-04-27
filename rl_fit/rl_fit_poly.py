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

from curve_normalization import PCA_normalization, fft_feature

from general_robotics_toolbox import *

sys.path.append('../greedy_fitting/')
from greedy_poly import greedy_fit

sys.path.append('../toolbox')
from robots_def import *

import warnings
warnings.filterwarnings("ignore")


# Util Class
Action = namedtuple("Action", ("Type", "Length", "Code"))
Primitive = namedtuple("Primitive", ("Start", "End", "Type", "Fits"))
Memory = namedtuple('Memory', ('state', 'action', 'reward', 'next_state', 'done'))


# Hyper-parameters
random_seed = 1234
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


def read_data():
    all_base = []
    all_js = []
    all_lam = []

    for i in range(201):
        lambda_file = "data/poly/lambda/lambda_{}.csv".format(i)
        base_file = "data/poly/base/curve_base_poly_{}.csv".format(i)
        js_file = "data/poly/js/curve_js_poly_{}.csv".format(i)

        lambda_data = pd.read_csv(lambda_file, header=None)
        lam = lambda_data.values[-1][0]

        col_names = ['poly_x', 'poly_y', 'poly_z', 'poly_direction_x', 'poly_direction_y', 'poly_direction_z']
        data = pd.read_csv(base_file, names=col_names)
        poly_x = data['poly_x'].tolist()
        poly_y = data['poly_y'].tolist()
        poly_z = data['poly_z'].tolist()
        curve_poly_coeff = np.vstack((poly_x, poly_y, poly_z))

        col_names = ['poly_q1', 'poly_q2', 'poly_q3', 'poly_q4', 'poly_q5', 'poly_q6']
        data = pd.read_csv(js_file, names=col_names)
        poly_q1 = data['poly_q1'].tolist()
        poly_q2 = data['poly_q2'].tolist()
        poly_q3 = data['poly_q3'].tolist()
        poly_q4 = data['poly_q4'].tolist()
        poly_q5 = data['poly_q5'].tolist()
        poly_q6 = data['poly_q6'].tolist()
        curve_js_poly_coeff = np.vstack((poly_q1, poly_q2, poly_q3, poly_q4, poly_q5, poly_q6))

        all_base.append(curve_poly_coeff)
        all_js.append(curve_js_poly_coeff)
        all_lam.append(lam)

    return all_base, all_js, all_lam


def save_data(episode_rewards, episode_steps, episode_target_curve):
    df = pd.DataFrame({"episode_rewards": episode_rewards,
                       "episode_steps": episode_steps,
                       "curve": episode_target_curve})
    df.to_csv("Training Curve Data.csv")


ERROR_THRESHOLD = 1.0
MAX_GREEDY_STEP = 10

EPS_START = 0.5
EPS_END = 0.01
EPS_DECAY = 10

TAU_START = 10
TAU_END = 0.01
MAX_TAU_EPISODE = 0.6
LEARNING_RATE = 0.001
LEARNING_FREQ = 4
GAMMA = 0.99
BATCH_SIZE = 256

REWARD_FINISH = 2000  # -------------------------------------- HERE
REWARD_DECAY_FACTOR = 0.9
REWARD_STEP = -10

SAVE_MODEL = 100


def reward_function(i_step, done):
    reward = REWARD_STEP
    if done:
        reward += REWARD_FINISH * (REWARD_DECAY_FACTOR ** i_step)

    return reward


class MemoryReplayer(object):
    def __init__(self, capacity=int(1e6)):
        self.memory = deque([], maxlen=capacity)

    def push(self, memory):
        state, action, reward, next_state, done = memory
        state = state.T
        if next_state is not None:
            next_state = next_state.T
        else:
            next_state = np.zeros_like(state)
        new_memory = Memory(state, action, reward, next_state, done)
        self.memory.append(new_memory)

    def sample(self, batch_size=BATCH_SIZE):
        sample_batch = random.sample(self.memory, batch_size)
        state_batch = np.stack([m.state for m in sample_batch])
        action_batch = np.array([m.action for m in sample_batch])
        reward_batch = np.array([m.reward for m in sample_batch])
        next_state_batch = np.stack([m.next_state for m in sample_batch])
        done_batch = np.array([m.done for m in sample_batch])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(3000, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.output(x)
        return x


class Agent(object):
    def __init__(self, n_action: int):
        self.n_action = n_action
        self.output_dim = n_action * 2
        self.lr = LEARNING_RATE
        self.gamma = GAMMA
        self.memory = MemoryReplayer()
        self.batch_size = BATCH_SIZE

        self.target_net = DQN(output_dim=self.output_dim)
        self.policy_net = DQN(output_dim=self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict().copy())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_function = nn.SmoothL1Loss()

        self.action_types = ['movel_fit'] * n_action + ['movec_fit'] * n_action + ['movej_fit'] * n_action

        self.best_reward = -9999
        self.roll_back_counter = 0
        self.best_policy = None

    def get_action(self, state, epsilon=0.):
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state.T)
            state_tensor = state_tensor.reshape(1, state_tensor.shape[0], state_tensor.shape[1])

            eps_sample = np.random.rand()
            if eps_sample < epsilon:
                action_code = np.random.randint(0, self.output_dim)
            else:
                output = self.policy_net(state_tensor.float())
                action_code = torch.argmax(output).data.detach().numpy()

            primitive_type = self.action_types[action_code]
            primitive_length = ((action_code % self.n_action) + 1) * (1 / self.n_action)
            action = Action(Type=primitive_type, Length=primitive_length, Code=action_code)
            return action

    def memory_save(self, state, action, reward, next_state, done):
        action_code = action.Code
        memory = Memory(state, action_code, reward, next_state, done)
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
        loss = self.loss_function(td_est, td_tgt)
        loss.backward()
        self.optimizer.step()

    def roll_back(self, episode_rewards):
        if len(episode_rewards) < 2000:
            return

        running_reward = np.mean(episode_rewards[-200:])
        if running_reward >= self.best_reward:
            self.best_reward = running_reward
            self.roll_back_counter = 0
            self.best_policy = self.policy_net.state_dict().copy()
            return
        else:
            self.roll_back_counter += 1

        if self.roll_back_counter > 1000:
            print("*** [Rollback] Episode {}; Best Reward {:2f}".format(len(episode_rewards), self.best_reward))
            self.policy_net.load_state_dict(self.best_policy.copy())
            self.target_net.load_state_dict(self.best_policy.copy())
            self.roll_back_counter = 0

    def update_target_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + os.sep + 'DQN_policy_net.pth')

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path + os.sep + 'DQN_policy_net.pth'))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class TrajEnv(greedy_fit):
    def __init__(self, robot, curve_poly_coeff, curve_js_poly_coeff, lam_f=1758.276831, num_points=50000,
                 orientation_weight=1):

        super().__init__(robot, curve_poly_coeff, curve_js_poly_coeff, lam_f=lam_f, num_points=num_points,
                         orientation_weight=orientation_weight)

        self.curve_fit = []
        self.curve_fit_R = []
        self.curve_fit_js = []
        self.cartesian_slope_prev = []
        self.rotation_axis_prev = []
        self.slope_diff = []
        self.slope_diff_ori = []
        self.js_slope_prev = None
        self.breakpoints = [0]
        self.fit_primitives = []

        self.greedy_primitives = {'movel_fit': None, 'movej_fit': None, 'movec_fit': None}

    def fit_primitive(self, max_error_threshold, max_ori_threshold=np.radians(3),
                      primitives=['movel_fit', 'movej_fit', 'movec_fit']):
        step_size = int(len(self.curve) / 20)
        ###initialize

        primitives_choices = []
        points = []

        next_point = min(step_size, len(self.curve) - self.breakpoints[-1])
        prev_point = 0
        prev_possible_point = 0

        max_errors = {'movel_fit': 999, 'movej_fit': 999, 'movec_fit': 999}
        max_ori_errors = {'movel_fit': 999, 'movej_fit': 999, 'movec_fit': 999}

        ###initial error map update:
        for key in primitives:
            curve_fit, curve_fit_R, curve_fit_js, max_error, max_ori_error = self.primitives[key](
                self.curve[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                self.curve_js[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                self.curve_R[self.breakpoints[-1]:self.breakpoints[-1] + next_point])
            max_errors[key] = max_error
            max_ori_errors[key] = max_ori_error

        ###bisection search self.breakpoints
        while True:
            # print('index: ', self.breakpoints[-1] + next_point, 'max error: ',
            #       max_errors[min(max_errors, key=max_errors.get)], 'max ori error (deg): ',
            #       np.degrees(max_ori_errors[min(max_ori_errors, key=max_ori_errors.get)]))
            ###bp going backward to meet threshold
            if min(list(max_errors.values())) > max_error_threshold or min(
                    list(max_ori_errors.values())) > max_ori_threshold:
                prev_point_temp = next_point
                next_point -= int(np.abs(next_point - prev_point) / 2)
                next_point = max(2, next_point)
                prev_point = prev_point_temp

                for key in primitives:
                    curve_fit, curve_fit_R, curve_fit_js, max_error, max_ori_error = self.primitives[key](
                        self.curve[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_js[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_R[self.breakpoints[-1]:self.breakpoints[-1] + next_point])
                    max_errors[key] = max_error
                    max_ori_errors[key] = max_ori_error

            ###bp going forward to get close to threshold
            else:
                prev_possible_point = next_point
                prev_point_temp = next_point
                next_point = min(next_point + int(np.abs(next_point - prev_point)),
                                 len(self.curve) - self.breakpoints[-1])
                next_point = max(2, next_point)
                prev_point = prev_point_temp

                for key in primitives:
                    curve_fit, curve_fit_R, curve_fit_js, max_error, max_ori_error = self.primitives[key](
                        self.curve[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_js[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_R[self.breakpoints[-1]:self.breakpoints[-1] + next_point])
                    max_errors[key] = max_error
                    max_ori_errors[key] = max_ori_error

            # print(max_errors)
            if next_point == prev_point:
                # print('stuck, restoring previous possible index')  ###if ever getting stuck, restore
                next_point = max(prev_possible_point, 2)
                # if self.breakpoints[-1]+next_point+1==len(self.curve)-1:
                # 	next_point=3

                primitives_added = False
                for key in primitives:
                    curve_fit, curve_fit_R, curve_fit_js, max_error, max_ori_error = self.primitives[key](
                        self.curve[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_js[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_R[self.breakpoints[-1]:self.breakpoints[-1] + next_point])
                    if max_error < max_error_threshold:
                        primitives_added = True
                        primitives_choices.append(key)
                        if key == 'movec_fit':
                            points.append([curve_fit[int(len(curve_fit) / 2)], curve_fit[-1]])
                        elif key == 'movel_fit':
                            points.append([curve_fit[-1]])
                        else:
                            points.append([curve_fit_js[-1]])
                        break
                if not primitives_added:
                    curve_fit, curve_fit_R, curve_fit_js, max_error, max_ori_error = self.movej_fit(
                        self.curve[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_js[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_R[self.breakpoints[-1]:self.breakpoints[-1] + next_point])
                    # print('primitive skipped1')
                    primitives_choices.append('movej_fit')
                    points.append([curve_fit_js[-1]])

                break

            # find the closest but under max_threshold
            if min(list(max_errors.values())) <= max_error_threshold and min(
                    list(max_ori_errors.values())) <= max_ori_threshold and np.abs(next_point - prev_point) < 10:
                primitives_added = False
                for key in primitives:
                    curve_fit, curve_fit_R, curve_fit_js, max_error, max_ori_error = self.primitives[key](
                        self.curve[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_js[self.breakpoints[-1]:self.breakpoints[-1] + next_point],
                        self.curve_R[self.breakpoints[-1]:self.breakpoints[-1] + next_point])
                    if max_error < max_error_threshold:
                        primitives_added = True
                        primitives_choices.append(key)
                        if key == 'movec_fit':
                            points.append([curve_fit[int(len(curve_fit) / 2)], curve_fit[-1]])
                        elif key == 'movel_fit':
                            points.append([curve_fit[-1]])
                        else:
                            points.append([curve_fit_js[-1]])
                        break
                if not primitives_added:
                    # print('primitive skipped2')
                    primitives_choices.append('movel_fit')
                    points.append([curve_fit[-1]])

                break

        if self.break_early:
            idx = max(
                self.breakearly(self.curve_backproj[self.breakpoints[-1]:self.breakpoints[-1] + next_point], curve_fit),
                2)
        else:
            idx = next_point

        next_break_point = min(self.breakpoints[-1] + idx, len(self.curve))

        # print(self.breakpoints)
        # print(primitives_choices)
        curve_fit = curve_fit[:len(curve_fit) - (next_point-idx)]
        curve_fit_R = curve_fit_R[:len(curve_fit) - (next_point-idx)]
        if primitives_choices[-1] == 'movej_fit':
            curve_fit_js = curve_fit_js[:len(curve_fit) - (next_point - idx)]

        return (self.breakpoints[-1], next_break_point), primitives_choices, (curve_fit, curve_fit_R, curve_fit_js)

    def get_greedy_primitives(self):

        for key in self.primitives:
            breakpoints, primitive_type, fits = self.fit_primitive(0.5, primitives=[key])
            start_point, end_point = breakpoints
            # assert (key == primitive_type[0])
            primitive_type = primitive_type[0]
            primitive = Primitive(start_point, end_point, primitive_type, fits)
            self.greedy_primitives[key] = primitive

        return self.greedy_primitives

    def reset(self):
        state = PCA_normalization(self.curve)
        done = False
        self.get_greedy_primitives()

        return state, done

    def step(self, action: Action, i_step):
        primitive_type = action.Type
        primitive_length = action.Length

        # Find the corresponding primitive based on Action
        greedy_primitive = self.greedy_primitives[primitive_type]
        new_primitive_length = np.ceil((greedy_primitive.End - greedy_primitive.Start) * primitive_length).astype(int)
        curve_fit, curve_fit_R, curve_fit_js = greedy_primitive.Fits
        new_breakpoint = greedy_primitive.Start + new_primitive_length
        new_breakpoint = min(new_breakpoint, len(self.curve))
        new_primitive_length = new_breakpoint - greedy_primitive.Start

        # ============== Add the fitting curve ============ Greedy Part
        self.fit_primitives.append(greedy_primitive)
        self.breakpoints.append(new_breakpoint)
        self.curve_fit.extend(curve_fit[:new_primitive_length])
        self.curve_fit_R.extend(curve_fit_R[:new_primitive_length])
        if primitive_type == 'movej_fit':
            self.curve_fit_js.extend(curve_fit_js[:new_primitive_length])
        else:
            # inv here to save time
            self.curve_fit_js.extend(self.car2js(curve_fit[:new_primitive_length],
                                                 curve_fit_R[:new_primitive_length]))

        # calculating ending slope
        R_diff = np.dot(curve_fit_R[0].T, curve_fit_R[new_primitive_length - 1])
        k, theta = R2rot(R_diff)
        if len(self.cartesian_slope_prev) > 0:
            self.slope_diff.append(self.get_angle(self.cartesian_slope_prev, curve_fit[1] - curve_fit[0]))
            self.slope_diff_ori.append(self.get_angle(self.rotation_axis_prev, k))

        if len(self.curve_fit) > 1:
            self.rotation_axis_prev = k
            self.cartesian_slope_prev = (self.curve_fit[-1] - self.curve_fit[-2]) / np.linalg.norm(
                self.curve_fit[-1] - self.curve_fit[-2])
            self.js_slope_prev = (self.curve_fit_js[-1] - self.curve_fit_js[-2]) / np.linalg.norm(
                self.curve_fit_js[-1] - self.curve_fit_js[-2])
            self.q_prev = self.curve_fit_js[-1]

        # =======================

        # ============ RL Part ===============
        done = self.breakpoints[-1] >= len(self.curve) - 1
        reward = reward_function(i_step, done)
        state = None
        if not done:
            last_breakpoint = self.breakpoints[-1]
            remaining_curve = self.curve[last_breakpoint:, :]
            normalized_curve = PCA_normalization(remaining_curve)
            state = normalized_curve
            self.get_greedy_primitives()
        else:
            self.curve_fit = np.array(self.curve_fit)
            self.curve_fit_R = np.array(self.curve_fit_R)
            self.curve_fit_js = np.array(self.curve_fit_js)
        # print(self.breakpoints, reward, done)

        return state, reward, done

    def plot_curve(self, curve_idx):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(self.curve[:, 0], self.curve[:, 1], self.curve[:, 2], 'red', linewidth=10)
        ax.scatter3D(self.curve_fit[:, 0], self.curve_fit[:, 1], self.curve_fit[:, 2], c=self.curve_fit[:,2], cmap='Greens')
        plt.title("#BP {}; #Primitive {}; Length {}".format(len(self.breakpoints), len(self.fit_primitives), len(self.curve_fit)))
        plt.savefig("plots/poly_rl/{}.jpg".format(curve_idx))
        plt.close()
        # plt.show()


def train(agent: Agent, base_data, js_data, lam_data, max_episode=20000):
    robot = abb6640(d=50)

    print("RL Training Start")

    episode_rewards = []
    episode_steps = []
    episode_target_curves = []

    start_learn = 200

    for i_episode in range(max_episode):
        timer = time.time()
        epsilon = EPS_START - min(1., max(0, i_episode - start_learn) / (max_episode * 0.8)) * (EPS_START - EPS_END)
        epsilon = 1. if i_episode < start_learn else epsilon
        episode_reward = 0
        target_curve_idx = np.random.randint(0, len(base_data))
        curve_poly_coeff = base_data[target_curve_idx]
        curve_js_coeff = js_data[target_curve_idx]
        lam = lam_data[target_curve_idx]

        env = TrajEnv(robot, curve_poly_coeff, curve_js_coeff, lam_f=lam, num_points=500)
        episode_memory = []
        state, done = env.reset()

        i_step = 0
        while not done:
            i_step += 1
            action = agent.get_action(state, epsilon)
            next_state, reward, done = env.step(action, i_step)
            episode_memory.append((state, action, reward, next_state, done))
            episode_reward += reward
            if done:
                break
            state = next_state

        for i, (state, action, reward, next_state, done) in enumerate(episode_memory):
            new_reward = reward + REWARD_DECAY_FACTOR ** (i_step - i) * episode_reward
            agent.memory_save(state=state, action=action, reward=new_reward, next_state=next_state, done=done)

        if i_episode > start_learn:
            for i in range(10):
                agent.learn()
            agent.update_target_nets()

        episode_rewards.append(episode_reward)
        episode_steps.append(len(env.fit_primitives))
        episode_target_curves.append(target_curve_idx)

        print("Episode {} / {} Epsilon: {:.3f} --- {} Steps --- Reward: {:.3f} --- {:.3f}s Curve {}"
              .format(i_episode + 1, max_episode, epsilon, i_step, episode_reward,
                      time.time() - timer, target_curve_idx))
        agent.roll_back(episode_rewards)

        if (i_episode + 1) % SAVE_MODEL == 0:
            print("=== Saving Model ===")
            save_data(episode_rewards, episode_steps, episode_target_curves)
            agent.save_model('model')
            print("DQNs saved at '{}'".format('model/'))
            print("======")

        env.plot_curve(target_curve_idx)


def main():
    base_data, js_data, lam_data = read_data()
    agent = Agent(n_action=10)
    train(agent, base_data, js_data, lam_data)


if __name__ == '__main__':
    main()
