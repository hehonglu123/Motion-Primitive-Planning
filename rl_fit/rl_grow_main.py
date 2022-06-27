import os.path

import numpy as np
import torch
import pandas as pd
import argparse
import time

from MotionPrimitive import MotionPrimitiveEnv
from rl_agent import DQNAgent
from replayer import Replayer
from toolbox.robots_def import *
from general_robotics_toolbox import *

import warnings
warnings.filterwarnings("ignore")


class IndexLoader(object):
    def __init__(self, max_idx):
        self.max_idx = max_idx
        self.index = np.arange(max_idx)
        self.pointer = 0

        self.reset()

    def reset(self):
        np.random.shuffle(self.index)
        self.pointer = 0

    def next(self):
        ret = self.index[self.pointer]
        self.pointer += 1
        if self.pointer >= self.max_idx:
            self.reset()
        return ret


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_error_threshold", type=int, default=0.5)
    parser.add_argument("--max_ori_threshold", type=float, default=np.radians(3))
    parser.add_argument("--replayer_capacity", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_train_episode", type=int, default=int(1e5))
    parser.add_argument("--warm_up", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--hidden_width", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=21)
    parser.add_argument("--discrete_actions", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num_normalize_points", type=int, default=1000)
    parser.add_argument("--learn_epochs", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=2000)

    args = parser.parse_args()
    return args


def train(curve_js_data, args):
    agent = DQNAgent(args)
    replayer = Replayer(args)
    curve_loader = IndexLoader(len(curve_js_data))
    timer = time.time()
    epsilon = 0.5

    for episode in range(args.max_train_episode):
        curve_js = curve_js_data[curve_loader.next()]
        env = MotionPrimitiveEnv(curve_js, args)
        episode_reward = 0
        is_warm_up = episode < args.warm_up
        if not is_warm_up:
            epsilon -= args.eps_decay

        state, done = env.reset(), False
        i_step = 0
        episode_memory = []
        while not done:
            i_step += 1
            if is_warm_up:
                action = np.random.randint(args.action_dim)
            else:
                if np.random.uniform() < epsilon:
                    action = np.random.randint(args.action_dim)
                else:
                    action = agent.choose_action(state)
            next_state, reward, done = env.rl_step(action)
            episode_memory.append((state, action, reward, next_state, done))
            episode_reward += reward
            state = next_state

        for i, (state, action, reward, next_state, done) in enumerate(episode_memory):
            step_length = (env.breakpoints[i+1] - env.breakpoints[i]) / env.breakpoints[-1]
            new_reward = episode_reward * (1 - step_length)
            replayer.store(state, action, new_reward, next_state, done)

        if not is_warm_up and replayer.size > args.batch_size:
            for i in range(args.learn_epochs):
                agent.learn(replayer)

        print("[Train]\t{:>7}/{:>7}\tAvg Time {:.2f}".format(episode+1, args.max_train_episode, (time.time()-timer)/(episode+1)))

        if (episode + 1) % args.eval_freq == 0:
            eval_df = evaluation(agent, curve_js_data, args)
            if not os.path.isdir('DQN_result'):
                os.mkdir('DQN_result')
            eval_df.to_csv('DQN_result/eval_{}.csv'.format(episode+1))
            if not os.path.isdir('DQN_result/model/eval_{}'.format(episode+1)):
                os.mkdir('DQN_result/model/eval_{}'.format(episode+1))
            agent.save_model('DQN_result/model/eval_{}'.format(episode+1))


def evaluation(agent: DQNAgent, curve_js_data, args):
    eval_df = {'idx': [], 'reward':[], 'num_primitive': [], 'exec_time': [], 'error': [], 'ori_error': []}

    for i, curve_js in curve_js_data:
        env = MotionPrimitiveEnv(curve_js, args)
        episode_reward = 0
        state, done = env.reset(), False
        step = 0

        while not done:
            step += 1
            action = agent.evaluate(state)
            next_state, reward, done = env.rl_step(action)
            episode_reward += reward
            state = next_state

        print("[Eval]\t{:>5}/{:>5}\tReward {:>5}\tExec Time {:.3f>7}\tError {:.3f>7}".format(i, len(curve_js_data), episode_reward, env.exec_time, env.max_error))

        eval_df['idx'] = i
        eval_df['reward'] = episode_reward
        eval_df['num_primitive'] = step
        eval_df['exec_time'] = env.exec_time
        eval_df['error'] = env.max_error
        eval_df['ori_error'] = env.max_ori_error

    return pd.DataFrame(eval_df)



def main():
    curve_js_data = read_data()
    args = get_args()
    args.robot = abb6640(d=50)
    args.action_dim = args.discrete_actions * 3
    args.curve_dim = args.num_normalize_points * 3
    args.device = torch.device('cpu')
    args.eps_decay = 0.5 / (args.max_train_episode - args.warm_up)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(curve_js_data, args)


if __name__ == '__main__':
    main()





