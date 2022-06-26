import torch
import numpy as np


class Replayer(object):
    def __init__(self, args):

        self.max_size = args.replayer_capacity
        self.size = 0
        self.pointer = 0
        self.device = args.device

        self.state_curve = np.zeros((self.max_size, args.curve_dim))
        self.state_feature = np.zeros((self.max_size, args.feature_dim))
        self.action = np.zeros((self.max_size, args.action_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.next_state_curve = np.zeros((self.max_size, args.curve_dim))
        self.next_state_feature = np.zeros((self.max_size, args.feature_dim))
        self.done = np.zeros((self.max_size, 1))

    def store(self, state, action, reward, next_state, done):
        self.state_curve[self.pointer] = state[0].flatten()
        self.state_feature[self.pointer] = state[1]
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state_curve[self.pointer] = next_state[0].flatten() if next_state is not None else 0
        self.next_state_feature[self.pointer] = next_state[1] if next_state is not None else 0
        self.done[self.pointer] = float(done)

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        batch_idx = np.random.choice(np.arange(self.size), size=batch_size, replace=False)
        return (
            torch.FloatTensor(self.state_curve[batch_idx]).to(self.device),
            torch.FloatTensor(self.state_feature[batch_idx]).to(self.device),
            torch.LongTensor(self.action[batch_idx]).to(self.device),
            torch.FloatTensor(self.reward[batch_idx]).to(self.device),
            torch.FloatTensor(self.next_state_curve[batch_idx]).to(self.device),
            torch.FloatTensor(self.next_state_feature[batch_idx]).to(self.device),
            torch.FloatTensor(self.done[batch_idx]).to(self.device)
        )


class PriorityReplayer(Replayer):
    def __init__(self, args):
        super(PriorityReplayer, self).__init__(args)

        self.weights = np.zeros(self.max_size)
        self.max_weight = 1e-2
        self.weight_offset = 1e-4
        self.batch_idx = None

    def store(self, state, action, reward, next_state, done):
        self.state_curve[self.pointer] = state[0]
        self.state_feature[self.pointer] = state[1]
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state_curve[self.pointer] = next_state[0]
        self.next_state_feature[self.pointer] = next_state[1]
        self.done[self.pointer] = float(done)
        self.weights[self.pointer] = self.max_weight

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        memory_weights = self.weights[:self.size] + self.weight_offset
        memory_probs = memory_weights / sum(memory_weights)
        self.batch_idx = np.random.choice(np.arange(self.size), size=batch_size, p=memory_probs, replace=False)
        return (
            torch.FloatTensor(self.state_curve[self.batch_idx]).to(self.device),
            torch.FloatTensor(self.state_feature[self.batch_idx]).to(self.device),
            torch.LongTensor(self.action[self.batch_idx]).to(self.device),
            torch.FloatTensor(self.reward[self.batch_idx]).to(self.device),
            torch.FloatTensor(self.next_state_curve[self.batch_idx]).to(self.device),
            torch.FloatTensor(self.next_state_feature[self.batch_idx]).to(self.device),
            torch.FloatTensor(self.done[self.batch_idx]).to(self.device)
        )

    def update_weights(self, td_error):
        max_error = np.max(td_error)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.batch_idx] = td_error.reshape(-1)
