import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from trajectory_utils import running_reward


def plot_running_reward(episode_rewards, n=200, show=True, save=True):
    rewards = running_reward(episode_rewards, n)
    fig = plt.figure()
    x = np.linspace(1, len(rewards), len(rewards))
    plt.plot(x, rewards)
    plt.grid()
    plt.title('Training Rewards')
    if save:
        plt.savefig('plots/running_reward.jpg')
    if show:
        plt.show()


def plot_rl_greedy_ratio(rl_steps, rl_curve_idx, greedy_steps, show=True, save=True):
    ratios = []
    mean_ratios = []
    median_ratios = []
    for i in range(len(rl_steps)):
        target_curve_idx = rl_curve_idx[i]
        greedy_step = greedy_steps[target_curve_idx]
        rl_step = rl_steps[i]
        ratios.append(greedy_step/rl_step)
        mean_ratio = np.mean(ratios[max(0, i-100):i+1])
        mean_ratios.append(mean_ratio)
        median = np.median(ratios[max(0, i-100):i+1])
        median_ratios.append(median)

    fig = plt.figure()
    x = np.linspace(1, len(rl_steps), len(rl_steps))
    plt.plot(x, ratios, color='blue', label='ratio')
    plt.plot(x, mean_ratios, color='red', label='mean_100')
    plt.plot(x, median_ratios, color='green', label='median_100')
    plt.legend()
    plt.grid()
    plt.title('Greedy/RL Ratio')
    if save:
        plt.savefig('plots/rl_greedy_ratio_train.jpg')
    if show:
        plt.show()


def plot_curve(curve, other_curves=None, show=True, save=False, save_path=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2], color='red', linewidth=3)

    if other_curves is not None:
        for c in other_curves:
            ax.plot3D(c[:, 0], c[:, 1], c[:, 2], 'green')

    if save:
        if save_path is None:
            plt.savefig('plots/temp_plot.jpg')
        else:
            plt.savefig(save_path)
    if show:
        plt.show()


if __name__ == '__main__':
    data = 'Training Curve data.csv'
    rl_df = pd.read_csv(data)
    episode_rewards = rl_df['episode_rewards']
    plot_running_reward(episode_rewards, save=False, show=True)

    greedy_data = 'data/poly_greedy_data.csv'
    greedy_df = pd.read_csv(greedy_data)
    greedy_steps_data = greedy_df['n_primitives']
    rl_steps_data = rl_df['episode_steps']
    rl_target_curve = rl_df['curve']
    plot_rl_greedy_ratio(rl_steps_data, rl_target_curve, greedy_steps_data, save=False, show=True)
