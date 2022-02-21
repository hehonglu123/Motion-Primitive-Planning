import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from trajectory_utils import running_reward


def plot_running_reward(episode_rewards, n=100):
    rewards = running_reward(episode_rewards, n)
    fig = plt.figure()
    x = np.linspace(1, len(rewards), len(rewards))
    plt.plot(x, rewards)
    plt.savefig('plots/running_reward.jpg')


def plot_rl_greedy_ratio(rl_steps, rl_curve_idx, greedy_steps):
    ratios = []
    for i in range(len(rl_steps)):
        target_curve_idx = rl_curve_idx[i]
        greedy_step = greedy_steps[target_curve_idx]
        rl_step = rl_steps[i]
        ratios.append(greedy_step/rl_step)

    fig = plt.figure()
    x = np.linspace(1, len(rl_steps), len(rl_steps))
    plt.plot(x, ratios)
    plt.savefig('plots/rl_greedy_ratio_train.jpg')


if __name__ == '__main__':
    data = 'Training Curve data.csv'
    rl_df = pd.read_csv(data)
    episode_rewards = rl_df['episode_rewards']
    plot_running_reward(episode_rewards)

    greedy_data = 'data/greedy_data.csv'
    greedy_df = pd.read_csv(greedy_data)
    greedy_steps_data = greedy_df['n_primitives']
    rl_steps_data = rl_df['episode_steps']
    rl_target_curve = rl_df['curve']
    plot_rl_greedy_ratio(rl_steps_data, rl_target_curve, greedy_steps_data)
