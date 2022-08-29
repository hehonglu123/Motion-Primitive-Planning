import numpy as np
import os
import torch


class Replayer(object):
    def __init__(self, args, capacity=int(1e5)):
        self.capacity = capacity
        self.size = 0
        self.pointer = 0

        self.state_curve = np.zeros((capacity, args.curve_normalize_dim*3*2))
        self.state_feature = np.zeros((capacity, args.state_robot_dim))

        self.next_state_curve = np.zeros((capacity, args.curve_normalize_dim*3*2))
        self.next_state_feature = np.zeros((capacity, args.state_robot_dim))

        self.action = np.zeros((capacity, args.action_dim))
        self.reward = np.zeros((capacity, 1))
        self.done = np.zeros((capacity, 1))
        self.success = np.zeros((capacity, 1))

    def store(self, state, action, reward, next_state, done, success):
        curve_error, curve_target, robot, is_start, is_end = state
        next_curve_error, next_curve_target, next_robot, next_is_start, next_is_end = next_state
        data_size = len(curve_error)

        for i in range(data_size):
            self.state_curve[self.pointer, :] = np.hstack([curve_error[i].flatten(), curve_target[i].flatten()])
            self.state_feature[self.pointer, :] = np.hstack([robot[i], is_start[i], is_end[i]])
            self.action[self.pointer, :] = action[i, :]
            self.reward[self.pointer, :] = reward[i]
            self.next_state_curve[self.pointer, :] = np.hstack([next_curve_error[i].flatten(), next_curve_target[i].flatten()])
            self.next_state_feature[self.pointer, :] = np.hstack([next_robot[i], next_is_start[i], next_is_end[i]])
            self.done[self.pointer, :] = done[i]
            self.success[self.pointer, :] = success[i]

            self.pointer += 1
            self.pointer = 0 if self.pointer >= self.capacity else self.pointer
            self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        sample_idx = np.random.randint(0, self.size, batch_size)

        return (torch.FloatTensor(self.state_curve[sample_idx, :]),
                torch.FloatTensor(self.state_feature[sample_idx, :]),
                torch.FloatTensor(self.action[sample_idx, :]),
                torch.FloatTensor(self.reward[sample_idx, :]),
                torch.FloatTensor(self.next_state_curve[sample_idx, :]),
                torch.FloatTensor(self.next_state_feature[sample_idx, :]),
                torch.LongTensor(self.done[sample_idx, :]),
                torch.LongTensor(self.success[sample_idx, :]))

    def save_to_file(self, dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        with open(dir_path + os.sep + 'state_curve.npy', 'wb') as f:
            np.save(f, self.state_curve)

        with open(dir_path + os.sep + 'state_feature.npy', 'wb') as f:
            np.save(f, self.state_feature)

        with open(dir_path + os.sep + 'action.npy', 'wb') as f:
            np.save(f, self.action)

        with open(dir_path + os.sep + 'reward.npy', 'wb') as f:
            np.save(f, self.reward)

        with open(dir_path + os.sep + 'next_state_curve.npy', 'wb') as f:
            np.save(f, self.next_state_curve)

        with open(dir_path + os.sep + 'next_state_feature.npy', 'wb') as f:
            np.save(f, self.next_state_feature)

        with open(dir_path + os.sep + 'done.npy', 'wb') as f:
            np.save(f, self.done)

        with open(dir_path + os.sep + 'success.npy', 'wb') as f:
            np.save(f, self.success)

    def load(self, dir_path):
        state_curve = np.load(dir_path + os.sep + 'state_curve.npy')
        state_feature = np.load(dir_path + os.sep + 'state_feature.npy')
        action = np.load(dir_path + os.sep + 'action.npy')
        reward = np.load(dir_path + os.sep + 'reward.npy')
        next_state_curve = np.load(dir_path + os.sep + 'next_state_curve.npy')
        next_state_feature = np.load(dir_path + os.sep + 'next_state_feature.npy')
        done = np.load(dir_path + os.sep + 'done.npy')
        success = np.load(dir_path + os.sep + 'success.npy')

        print("[Replayer Memory]: File has {} memory".format(state_curve.shape[0]))
        memory_size = min(self.capacity, state_curve.shape[0])

        self.state_curve[:memory_size, :] = state_curve[-memory_size:, :]
        self.state_feature[:memory_size, :] = state_feature[-memory_size:, :]
        self.action[:memory_size, :] = action[-memory_size:, :]
        self.reward[:memory_size, :] = reward[-memory_size:, :]
        self.next_state_curve[:memory_size, :] = next_state_curve[-memory_size:, :]
        self.next_state_feature[:memory_size, :] = next_state_feature[-memory_size:, :]
        self.done[:memory_size, :] = done[-memory_size:, :]
        self.success[:memory_size, :] = success[-memory_size:, :]

        self.size = memory_size
        self.pointer = memory_size % self.capacity

        print("[Replayer Memory]: load {} memory".format(memory_size))

