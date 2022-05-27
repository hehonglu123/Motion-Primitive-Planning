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
from greedy_new import greedy_fit

sys.path.append('../toolbox')
from robots_def import *
from general_robotics_toolbox import *
from MotionSend import *

import warnings
warnings.filterwarnings("ignore")


# Util Class
Action = namedtuple("Action", ("Type", "Length", "Code"))
Primitive = namedtuple("Primitive", ("Start", "End", "Type", "Fits", "Error", "OriError"))
Memory = namedtuple('Memory', ('state', 'action', 'reward', 'next_state', 'done'))


# Hyper-parameters
random_seed = 1234
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


def read_data():
    # all_base = []
    all_js = []

    for i in range(201):
        # base_file = "data/poly/base/curve_base_poly_{}.csv".format(i)
        js_file = "data/js_new_500/traj_{}_js_new.csv".format(i)

        col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6']
        data = pd.read_csv(js_file, names=col_names)
        curve_q1 = data['q1'].tolist()
        curve_q2 = data['q2'].tolist()
        curve_q3 = data['q3'].tolist()
        curve_q4 = data['q4'].tolist()
        curve_q5 = data['q5'].tolist()
        curve_q6 = data['q6'].tolist()
        curve_js = np.vstack((curve_q1, curve_q2, curve_q3, curve_q4, curve_q5, curve_q6)).T

        # all_base.append(curve_poly_coeff)
        all_js.append(curve_js)

    return all_js


EPS_START = 0.5
EPS_END = 0.0001
EPS_DECAY = 10

LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 256
MEM_CAP = 1e6

REWARD_FINISH = 2000  # -------------------------------------- HERE
REWARD_DECAY_FACTOR = 0.9
REWARD_STEP = -10
TIME_WEIGHT = 0.8
TIME_GAIN = 10
ERROR_WEIGHT = 1 - TIME_WEIGHT
ERROR_GAIN = 5

SAVE_MODEL = 100


class MemoryReplayer(object):
    def __init__(self, capacity=int(MEM_CAP)):
        self.capacity = capacity
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
        self.output_dim = n_action * 3
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
        self.best_episode = 0

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

    def roll_back(self, eval_reward, i_episode=0):
        reward = np.mean(eval_reward)
        torch.save(self.policy_net.state_dict(), "model/checkpoints/model_{}.pth".format(i_episode))

        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_episode = i_episode
        else:
            print("*** [Rollback] Episode {}; Current Reward {}; Best Reward {:2f}".format(i_episode, reward, self.best_reward))
            self.policy_net.load_state_dict(torch.load("model/checkpoints/model_{}.pth".format(self.best_episode)))
            self.target_net.load_state_dict(torch.load("model/checkpoints/model_{}.pth".format(self.best_episode)))

    def update_target_nets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + os.sep + 'DQN_policy_net.pth')

    def load_model(self, path):
        # self.policy_net.load_state_dict(torch.load(path + os.sep + 'DQN_policy_net.pth'))
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class TrajEnv(greedy_fit):
    def __init__(self, robot, curve_js, orientation_weight=1):

        super().__init__(robot, curve_js, orientation_weight=orientation_weight)

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
        self.max_error = -1
        self.max_ori_error = -1

        self.primitive_choices = []
        self.points = []

        self.exec_time = -1
        self.exec_max_error = -1
        self.exec_max_error_angle = -1

        self.error_curve = None
        self.normal_error_curve = None

    def fit_primitive(self, max_error_threshold, max_ori_threshold=np.radians(3),
                      primitives=['movel_fit', 'movej_fit', 'movec_fit']):
        primitives_choices = []
        points = []

        next_point = min(20, len(self.curve) - self.breakpoints[-1])
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
                if primitives[0] == 'movec_fit':
                    next_point = max(3, next_point)
                else:
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
                if primitives[0] == 'movec_fit':
                    next_point = max(3, next_point)
                else:
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
                if primitives[0] == 'movec_fit':
                    next_point = max(prev_possible_point, 3)
                else:
                    next_point = max(prev_possible_point, 2)
                # next_point = max(prev_possible_point, 2)
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
        curve_fit = curve_fit[:len(curve_fit) - (next_point - idx)]
        curve_fit_R = curve_fit_R[:len(curve_fit) - (next_point - idx)]
        if primitives_choices[-1] == 'movej_fit':
            curve_fit_js = curve_fit_js[:len(curve_fit) - (next_point - idx)]

        return (self.breakpoints[-1], next_break_point), primitives_choices, (curve_fit, curve_fit_R, curve_fit_js), \
               (max_errors, max_ori_errors)

    def get_greedy_primitives(self):

        for key in self.primitives:
            breakpoints, primitive_type, fits, errors = self.fit_primitive(1., primitives=[key])
            start_point, end_point = breakpoints
            max_error, max_ori_error = errors
            # assert (key == primitive_type[0])
            primitive_type = primitive_type[0]
            primitive = Primitive(start_point, end_point, primitive_type, fits, max_error, max_ori_error)
            self.greedy_primitives[key] = primitive

        return self.greedy_primitives

    def execute_robot_studio(self, eval_mode=False):
        robot = abb6640(d=50)
        curve_js = self.curve_js
        breakpoints, primitives_choices, points = self.breakpoints, self.primitive_choices, self.points
        primitives_choices.insert(0, 'movej_fit')
        q_all = np.array(robot.inv(self.curve_fit[0], self.curve_fit_R[0]))
        ###choose inv_kin closest to previous joints
        temp_q = q_all - curve_js[0]
        order = np.argsort(np.linalg.norm(temp_q, axis=1))
        q_init = q_all[order[0]]
        points.insert(0, [q_init])

        act_breakpoints = np.array(breakpoints)

        act_breakpoints[1:] = act_breakpoints[1:] - 1

        #######################RS execution################################
        from io import StringIO
        ms = MotionSend()
        StringData = StringIO(
            ms.exec_motions(primitives_choices, act_breakpoints, points, self.curve_fit_js, v500, z10))
        df = pd.read_csv(StringData, sep=",")
        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R, speed, timestamp = ms.logged_data_analysis(df)

        if eval_mode:
            max_error, max_error_angle, max_error_idx, error_log, normal_error_log = calc_max_error_w_normal(curve_exe, self.curve,
                                                                                curve_exe_R[:, :, -1], self.curve_R[:, :, -1],
                                                                                eval_mode=eval_mode)
            self.error_curve = error_log
            self.normal_error_curve = normal_error_log
        else:
            max_error, max_error_angle, max_error_idx = calc_max_error_w_normal(curve_exe, self.curve,
                                                                                curve_exe_R[:, :, -1], self.curve_R[:, :, -1],
                                                                                eval_mode=eval_mode)

        # print('time: ', timestamp[-1] - timestamp[0], 'error: ', max_error, 'normal error: ', max_error_angle)
        execution_time = timestamp[-1] - timestamp[0]
        return execution_time, max_error, max_error_angle

    def reward_function(self, done, eval_mode=False):
        if not done:
            return 0
        else:
            execution_time, max_error, max_error_angle = self.execute_robot_studio(eval_mode=eval_mode)
            self.exec_time, self.exec_max_error, self.exec_max_error_angle = execution_time, max_error, max_error_angle
            reward = TIME_WEIGHT * TIME_GAIN * (3 - execution_time) \
                     - ERROR_WEIGHT * ERROR_GAIN * (max_error + max_error_angle)
            return reward

    def reset(self):
        state = PCA_normalization(self.curve)
        done = False
        self.get_greedy_primitives()

        return state, done

    def step(self, action: Action, i_step, eval_mode=False):
        primitive_type = action.Type
        primitive_length = action.Length

        # Find the corresponding primitive based on Action
        greedy_primitive = self.greedy_primitives[primitive_type]
        # assert greedy_primitive.Type == primitive_type
        new_primitive_length = np.ceil((greedy_primitive.End - greedy_primitive.Start) * primitive_length).astype(int)
        curve_fit, curve_fit_R, curve_fit_js = greedy_primitive.Fits
        new_breakpoint = greedy_primitive.Start + new_primitive_length
        new_breakpoint = min(new_breakpoint, len(self.curve))
        new_primitive_length = new_breakpoint - greedy_primitive.Start
        new_primitive_length = max(new_primitive_length, 2)
        self.max_error = max(self.max_error, greedy_primitive.Error[primitive_type])
        self.max_ori_error = max(self.max_ori_error, greedy_primitive.OriError[primitive_type])
        curve_fit = curve_fit[:new_primitive_length]
        curve_fit_R = curve_fit_R[:new_primitive_length]
        curve_fit_js = curve_fit_js[:new_primitive_length]

        # ============== Add the fitting curve ============ Greedy Part
        primitive_type = greedy_primitive.Type
        if primitive_type == 'movec_fit':
            if len(curve_fit) <= 5:
                self.points.append([curve_fit[-1]])
                primitive_type = 'movel_fit'
            else:
                self.points.append([curve_fit[int(len(curve_fit)/2)], curve_fit[-1]])
        elif primitive_type == 'movel_fit':
            self.points.append([curve_fit[-1]])
        else:
            self.points.append([curve_fit_js[-1]])
        self.primitive_choices.append(primitive_type)

        self.fit_primitives.append((primitive_type, curve_fit))
        self.breakpoints.append(new_breakpoint)
        self.curve_fit.extend(curve_fit)
        self.curve_fit_R.extend(curve_fit_R)
        if primitive_type == 'movej_fit':
            self.curve_fit_js.extend(curve_fit_js)
        else:
            # inv here to save time
            self.curve_fit_js.extend(self.car2js(curve_fit, curve_fit_R))

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
        reward = self.reward_function(done, eval_mode=eval_mode)
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

    def plot_curve(self, curve_idx=-1):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(self.curve[:, 0], self.curve[:, 1], self.curve[:, 2], 'red', linewidth=10)
        ax.scatter3D(self.curve_fit[:, 0], self.curve_fit[:, 1], self.curve_fit[:, 2], c=self.curve_fit[:,2], cmap='Greens')
        plt.title("#BP {}; #Primitive {}; Length {}; Error {:.3f}; OriError {:.3f}".format(len(self.breakpoints), len(self.fit_primitives), len(self.curve_fit), self.max_error, self.max_ori_error))
        plt.savefig("plots/poly_rl/{}.jpg".format(curve_idx), dpi=300)
        plt.close()
        # plt.show()

    def plot_primitives(self, curve_idx=-1):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(self.curve[:, 0], self.curve[:, 1], self.curve[:, 2], 'gray', linewidth=10, label='Target')

        used_types = {"movel_fit": False, "movec_fit": False, "movej_fit": False}

        for p_type, p_curve in self.fit_primitives:
            color = 'b'
            if p_type == 'movel_fit':
                color = 'b'
            elif p_type == 'movec_fit':
                color = 'r'
            elif p_type == 'movej_fit':
                color = 'g'
            if not used_types[p_type]:
                ax.plot3D(p_curve[:, 0], p_curve[:, 1], p_curve[:, 2], color=color, linewidth=2, label=p_type)
                used_types[p_type] = True
            else:
                ax.plot3D(p_curve[:, 0], p_curve[:, 1], p_curve[:, 2], color=color, linewidth=2)
            ax.plot3D(p_curve[0, 0], p_curve[0, 1], p_curve[0, 2], "mx", linewidth=3)
            # ax.plot3D(p_curve[-1, 0], p_curve[-1, 1], p_curve[-1, 2], "mx")


        plt.title("#BP {}; #Primitive {}; Length {}".format(len(self.breakpoints), len(self.fit_primitives),
                                                            len(self.curve_fit)))
        plt.legend()
        plt.savefig("plots/poly_rl_primitive/{}.jpg".format(curve_idx), dpi=300)
        plt.close()

    def plot_error(self, curve_idx=-1):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Point Index')
        ax1.set_ylabel('Error', color=color)
        ax1.plot(self.error_curve, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim((0, 5))

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Normal Error', color=color)
        ax2.plot(self.normal_error_curve, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim((0, 0.05))

        fig.tight_layout()
        plt.title("Curve {} - Error".format(curve_idx))
        plt.savefig("plots/rl_error/{}.jpg".format(curve_idx), dpi=300)
        plt.close()


def save_data(episode_rewards, episode_times, episode_errors, episode_error_angles, episode_target_curve, episode_steps):
    df = pd.DataFrame({"episode_rewards": episode_rewards,
                       "episode_times": episode_times,
                       "episode_errors": episode_errors,
                       "episode_error_angles": episode_error_angles,
                       "curve": episode_target_curve,
                       "steps": episode_steps})
    df.to_csv("Training Curve Data.csv")


def train(agent: Agent, js_data, max_episode=10000):
    robot = abb6640(d=50)

    print("RL Training Start")

    # ==== Load Previous Training Record ====
    # training_data = pd.read_csv("Training Curve Data.csv")
    # episode_rewards = [x for x in training_data["episode_rewards"].values]
    # episode_steps = [x for x in training_data["episode_steps"].values]
    # episode_target_curves = [x for x in training_data["curve"].values]
    # episode_start = len(episode_rewards)
    # =======================================

    # ==== Start from scratch ====
    episode_rewards = []
    episode_times = []
    episode_errors = []
    episode_error_angles = []
    episode_target_curves = []
    episode_steps = []
    episode_start = 0
    # ============================

    # ==== Evaluation Results ====
    eval_idxs = []
    eval_rewards = []
    eval_times = []
    eval_errors = []
    eval_error_angles = []
    eval_episodes = []
    # ============================

    start_learn = 500

    for i_episode in range(episode_start, max_episode):
        timer = time.time()
        epsilon = EPS_START - min(1., max(0, i_episode - start_learn) / (max_episode * 1.0)) * (EPS_START - EPS_END)
        epsilon = 1. if i_episode < start_learn * 2 else epsilon
        episode_reward = 0
        target_curve_idx = np.random.randint(0, len(js_data))
        curve_js = js_data[target_curve_idx]

        env = TrajEnv(robot, curve_js)
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

        # === Primitive Length Reward ===
        # for i, (state, action, reward, next_state, done) in enumerate(episode_memory):
        #     step_length = (env.breakpoints[i+1] - env.breakpoints[i]) / env.breakpoints[-1]
        #     # new_reward = reward + REWARD_DECAY_FACTOR ** (i_step - i) * episode_reward
        #     new_reward = episode_reward * (1 - step_length)
        #     agent.memory_save(state=state, action=action, reward=new_reward, next_state=next_state, done=done)
        # ===============================
        # === Primitive Slope Difference ===
        for i, (state, action, reward, next_state, done) in enumerate(episode_memory):
            slope_diff_1 = 0
            slope_diff_2 = 0
            primitive = env.fit_primitives[i][1]
            if i > 0:
                previous_primitive = env.fit_primitives[i-1][1]
                slope1 = primitive[1] - primitive[0]
                slope2 = previous_primitive[-1] - primitive[-2]
                slope_diff_1 = env.get_angle(slope2, slope1)
            if i < len(episode_memory) - 1:
                next_primitive = env.fit_primitives[i+1][1]
                slope1 = primitive[-1] - primitive[-2]
                slope2 = next_primitive[1] - next_primitive[0]
                slope_diff_2 = env.get_angle(slope1, slope2)
            new_reward = episode_reward * (slope_diff_1 + slope_diff_2) / 0.52
            agent.memory_save(state=state, action=action, reward=new_reward, next_state=next_state, done=done)
        # ==================================

        if i_episode > start_learn:
            for i in range(10):
                agent.learn()
            agent.update_target_nets()

        episode_rewards.append(episode_reward)
        episode_times.append(env.exec_time)
        episode_errors.append(env.exec_max_error)
        episode_error_angles.append(env.exec_max_error_angle)
        episode_target_curves.append(target_curve_idx)
        episode_steps.append(i_step)

        print("Episode {} / {} Epsilon: {:.3f} - Step: {} - Exec: {:.2f}s - Error: {:.2f} - Reward: {:.3f} - MemCap: {:.3f}% - {:.3f}s Curve {}"
              .format(i_episode + 1, max_episode, epsilon, i_step, env.exec_time, env.exec_max_error, episode_reward,
                      len(agent.memory)*100/agent.memory.capacity, time.time() - timer, target_curve_idx))
        # agent.roll_back(episode_rewards)

        if (i_episode + 1) % SAVE_MODEL == 0 and i_episode > start_learn:
            print("=== Saving Model ===")
            save_data(episode_rewards, episode_times, episode_errors, episode_error_angles,
                      episode_target_curves, episode_steps)
            # agent.save_model('model')
            print("DQNs saved at '{}'".format('model/'))
            print("======")

        if (i_episode + 1) % 2000 == 0 and i_episode > start_learn:
            eval_idx, eval_reward, eval_time, eval_error, eval_error_angle = evaluate(agent, js_data)
            eval_episode = [i_episode + 1] * len(eval_idx)
            eval_idxs += eval_idx
            eval_rewards += eval_reward
            eval_times += eval_time
            eval_errors += eval_error
            eval_error_angles += eval_error_angle
            eval_episodes += eval_episode

            eval_df = pd.DataFrame({"id": eval_idxs, "reward": eval_rewards, "time": eval_times, "error": eval_errors,
                                    "error_normal": eval_error_angles, "episode": eval_episodes})
            eval_df.to_csv("Train Eval Result.csv", index=False)
            agent.roll_back(eval_reward, i_episode + 1)

        # env.plot_curve(target_curve_idx)


def evaluate(agent: Agent, js_data):
    robot = abb6640(d=50)

    print("RL Evaluation Start")

    curve_rewards = []
    curve_times = []
    curve_errors = []
    curve_error_angles = []
    curve_idx = []

    for i in range(len(js_data)):

        timer = time.time()

        episode_reward = 0
        target_curve_idx = i
        curve_js = js_data[target_curve_idx]

        env = TrajEnv(robot, curve_js)
        state, done = env.reset()

        i_step = 0
        while not done:
            i_step += 1
            action = agent.get_action(state, 0.)
            next_state, reward, done = env.step(action, i_step, eval_mode=True)
            episode_reward += reward
            if done:
                break
            state = next_state

        print("[EVAL] Curve {}/{} --- Exec: {:.2f}s --- Error: {:.2f} --- Reward: {:.2f} --- {:.3f}s"
              .format(i, len(js_data), env.exec_time, env.exec_max_error, episode_reward, time.time() - timer))

        curve_rewards.append(episode_reward)
        curve_times.append(env.exec_time)
        curve_errors.append(env.exec_max_error)
        curve_error_angles.append(env.exec_max_error_angle)
        curve_idx.append(target_curve_idx)

        env.plot_curve(i)
        env.plot_primitives(i)
        env.plot_error(i)

    return curve_idx, curve_rewards, curve_times, curve_errors, curve_error_angles


def run_greedy():
    all_id = []
    all_time = []
    all_error = []
    all_normal_error = []

    for i in range(201):
        file_path = "data/js_new_500/traj_{}_js_new.csv".format(i)

        ###read in points
        curve_js = pd.read_csv(file_path, header=None).values

        robot = abb6640(d=50)

        greedy_fit_obj = greedy_fit(robot, curve_js, orientation_weight=1)

        breakpoints, primitives_choices, points = greedy_fit_obj.fit_under_error(0.5)

        ############insert initial configuration#################
        primitives_choices.insert(0, 'movej_fit')
        q_all = np.array(robot.inv(greedy_fit_obj.curve_fit[0], greedy_fit_obj.curve_fit_R[0]))
        ###choose inv_kin closest to previous joints
        temp_q = q_all - curve_js[0]
        order = np.argsort(np.linalg.norm(temp_q, axis=1))
        q_init = q_all[order[0]]
        points.insert(0, [q_init])

        act_breakpoints = np.array(breakpoints)

        act_breakpoints[1:] = act_breakpoints[1:] - 1

        print("========= {} / {} ==========".format(i, 201))

        #######################RS execution################################
        from io import StringIO
        ms = MotionSend()
        StringData = StringIO(
            ms.exec_motions(primitives_choices, act_breakpoints, points, greedy_fit_obj.curve_fit_js, v500, z10))
        df = pd.read_csv(StringData, sep=",")
        ##############################data analysis#####################################
        lam, curve_exe, curve_exe_R, speed, timestamp = ms.logged_data_analysis(df)
        max_error, max_error_angle, max_error_idx = calc_max_error_w_normal(curve_exe, greedy_fit_obj.curve,
                                                                            curve_exe_R[:, :, -1],
                                                                            greedy_fit_obj.curve_R[:, :, -1])

        print('time: ', timestamp[-1] - timestamp[0], 'error: ', max_error, 'normal error: ', max_error_angle)
        all_id.append(i)
        all_time.append(timestamp[-1] - timestamp[0])
        all_error.append(max_error)
        all_normal_error.append(max_error_angle)
        # break
        out_df = pd.DataFrame({"id": all_id, "time": all_time, "error": all_error, "normal_error": all_normal_error})
        out_df.to_csv("data/greedy_time_data.csv", index=False)


def main():
    # run_greedy()
    js_data = read_data()
    agent = Agent(n_action=10)
    # agent.load_model('model/checkpoints/model_6000.pth')
    train(agent, js_data)

    # eval_idx, eval_reward, eval_time, eval_error, eval_error_angle = evaluate(agent, js_data)
    # eval_episode = ['6000'] * len(eval_idx)
    # eval_df = pd.DataFrame({"id": eval_idx, "reward": eval_reward, "time": eval_time, "error": eval_error,
    #                         "error_normal": eval_error_angle, "episode": eval_episode})
    # eval_df.to_csv("Eval Result 6000.csv", index=False)

if __name__ == '__main__':
    main()
