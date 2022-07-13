import numpy as np
import torch


class Replayer(object):
    def __init__(self, args, capacity=int(1e5)):
        self.capacity = capacity
        self.size = 0
        self.pointer = 0

        self.state_curve = np.zeros((capacity, args.curve_normalize_dim*3*2))
        self.state_feature = np.zeros((capacity, args.state_robot_dim + 2))

        self.next_state_curve = np.zeros((capacity, args.curve_normalize_dim*3*2))
        self.next_state_feature = np.zeros((capacity, args.state_robot_dim + 2))

        self.action = np.zeros((capacity, args.action_dim))
        self.reward = np.zeros((capacity, 1))
        self.done = np.zeros((capacity, 1))

    def store(self, state, action, reward, next_state, done):
        curve_error, curve_target, robot, is_start, is_end = state
        next_curve_error, next_curve_target, next_robot, next_is_start, next_is_end = next_state
        data_size = len(curve_error)

        for i in range(data_size):
            self.state_curve[self.pointer, :] = np.hstack([curve_error[i].flatten(), curve_target[i].flatten()])
            self.state_feature[self.pointer, :] = np.hstack([robot[i], is_start[i], is_end[i]])
            self.action[self.pointer, :] = action
            self.reward[self.pointer, :] = reward
            self.next_state_curve[self.pointer, :] = np.hstack([next_curve_error[i].flatten(), next_curve_target[i].flatten()])
            self.next_state_feature[self.pointer, :] = np.hstack([next_robot[i], next_is_start[i], next_is_end[i]])
            self.done[self.pointer, :] = done

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
                torch.LongTensor(self.done[sample_idx, :]))
