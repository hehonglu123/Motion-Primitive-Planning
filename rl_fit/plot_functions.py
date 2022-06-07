import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from trajectory_utils import running_reward


def plot_evaluation(eval_data_file, greedy_steps, show=True, save=True):

    df = pd.read_csv(eval_data_file)
    curve_idx = df["id"].values
    n_primitive = df['n_primitive'].values

    ratios = []
    for i in range(len(curve_idx)):
        ratios.append(greedy_steps[curve_idx[i]] / n_primitive[i])

    df['ratio'] = ratios

    fig1 = sns.boxplot(x='episode', y='reward', data=df, color='royalblue')
    plt.title("Evaluation - Reward")
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Reward.jpg", dpi=300)
    if show:
        plt.show()

    fig2 = sns.boxplot(x='episode', y='ratio', data=df, color='orangered')
    plt.title("Evaluation - Greedy/RL Ratio")
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Ratio.jpg", dpi=300)
    if show:
        plt.show()


def plot_time_evaluation(eval_data_file, greedy_data_file, show=True, save=True):
    greedy_df = pd.read_csv(greedy_data_file)
    eval_df = pd.read_csv(eval_data_file)

    # eval_df = eval_df.loc[eval_df['episode'] == 6000]

    curve_idx = eval_df["id"].values
    exec_time = eval_df["time"].values
    error = eval_df["error"].values
    error_normal = eval_df["error_normal"].values

    time_ratio = []
    error_ratio = []
    error_normal_ratio = []

    for i in range(len(curve_idx)):
        time_ratio.append(greedy_df['time'].values[curve_idx[i]] / exec_time[i])
        error_ratio.append(greedy_df['error'].values[curve_idx[i]] / error[i])
        error_normal_ratio.append(greedy_df['normal_error'].values[curve_idx[i]] / error_normal[i])
    eval_df['time_ratio'] = time_ratio
    eval_df['error_ratio'] = error_ratio
    eval_df['error_normal_ratio'] = error_normal_ratio

    fig1 = sns.boxplot(x='episode', y='reward', data=eval_df, color='royalblue')
    plt.title("Evaluation - Reward")
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Reward.jpg", dpi=300)
    if show:
        plt.show()

    fig2 = sns.boxplot(x='episode', y='time', data=eval_df, color='royalblue')
    plt.title("Evaluation - Time")
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Time.jpg", dpi=300)
    if show:
        plt.show()

    fig3 = sns.boxplot(x='episode', y='error', data=eval_df, color='royalblue')
    plt.title("Evaluation - Error")
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Error.jpg", dpi=300)
    if show:
        plt.show()

    fig4 = sns.boxplot(x='episode', y='error_normal', data=eval_df, color='royalblue')
    plt.title("Evaluation - Normal Error")
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Normal_Error.jpg", dpi=300)
    if show:
        plt.show()

    fig5 = sns.boxplot(x='episode', y='time_ratio', data=eval_df, color='orangered')
    plt.title("Evaluation - Time Ratio")
    # plt.grid()
    # plt.ylim([0.9, 1.1])
    if save:
        plt.savefig("plots/Evaluation_Time_Ratio.jpg", dpi=300)
    if show:
        plt.show()

    fig6 = sns.boxplot(x='episode', y='error_ratio', data=eval_df, color='orangered')
    plt.title("Evaluation - Error Ratio")
    # plt.ylim([0.7, 1.5])
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Error_Ratio.jpg", dpi=300)
    if show:
        plt.show()

    fig7 = sns.boxplot(x='episode', y='error_normal_ratio', data=eval_df, color='orangered')
    plt.title("Evaluation - Error Normal Ratio")
    # plt.ylim([0.5, 2.0])
    # plt.grid()
    if save:
        plt.savefig("plots/Evaluation_Error_Normal_Ratio.jpg", dpi=300)
    if show:
        plt.show()




def plot_running_reward(episode_rewards, n=200, show=True, save=True):
    rewards = running_reward(episode_rewards, n)
    fig = plt.figure()
    x = np.linspace(1, len(rewards), len(rewards))
    plt.plot(x, rewards)
    plt.grid()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title('Training Rewards')
    if save:
        plt.savefig('plots/running_reward.jpg', dpi=300)
    if show:
        plt.show()


def plot_training_curve(episode_data, label, n=200, show=True, save=True):
    running_times = []
    for i in range(len(episode_data)):
        average_times = np.mean(episode_data[max(0, i-n):i+1])
        running_times.append(average_times)

    plt.figure()
    plt.plot(running_times)
    plt.grid()
    plt.xlabel("Episode")
    plt.ylabel(label)
    if save:
        plt.savefig('plots/{}.jpg'.format(label), dpi=300)
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
        mean_ratio = np.mean(ratios[max(0, i-200):i+1])
        mean_ratios.append(mean_ratio)
        median = np.median(ratios[max(0, i-200):i+1])
        median_ratios.append(median)

    fig = plt.figure()
    x = np.linspace(1, len(rl_steps), len(rl_steps))
    plt.plot(x, ratios, color='royalblue', label='ratio')
    plt.plot(x, mean_ratios, color='red', label='mean_100')
    plt.plot(x, median_ratios, color='green', label='median_100')
    plt.legend()
    plt.grid()
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.title('Greedy/RL Ratio')
    if save:
        plt.savefig('plots/rl_greedy_ratio_train.jpg', dpi=300)
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
    # data = 'Training Curve data.csv'
    # rl_df = pd.read_csv(data)
    # episode_rewards = rl_df['episode_rewards']
    # plot_running_reward(episode_rewards, save=True, show=True)
    #
    # greedy_data = 'data/poly_greedy_data.csv'
    # greedy_df = pd.read_csv(greedy_data)
    # greedy_steps_data = greedy_df['n_primitives']
    # rl_steps_data = rl_df['episode_steps']
    # rl_target_curve = rl_df['curve']
    # plot_rl_greedy_ratio(rl_steps_data, rl_target_curve, greedy_steps_data, save=True, show=True)
    #
    # plot_evaluation("Train Eval Result.csv", greedy_steps_data, show=True, save=True)

    data = "Training Curve data.csv"
    df = pd.read_csv(data)
    rewards = df['episode_rewards']
    times = df['episode_times']
    errors = df['episode_errors']
    error_angle = df['episode_error_angles']
    steps = df['steps']
    plot_training_curve(rewards, 'Reward', save=True, show=True)
    plot_training_curve(times, 'Execution Time', save=True, show=True)
    plot_training_curve(errors, 'Max Error', save=True, show=True)
    plot_training_curve(error_angle, 'Max Error Angle', save=True, show=True)
    plot_training_curve(steps, 'Steps', save=True, show=True)

    plot_time_evaluation("Train Eval Result.csv", "data/greedy_time_data.csv", show=True, save=True)

