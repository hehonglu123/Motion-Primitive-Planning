import os

import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt


def plot_curve():
    fig, ax = plt.subplots(subplot_kw={"projection": '3d'})
    for i in range(101, 102):
        curve = pd.read_csv('eval_data/curve1/forward/base/curve_{}.csv'.format(i), header=None).values
        ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2])
    plt.show()


def generate_gif(dir_path):

    # all_bp_image = []
    # all_profile_image = []

    num_frame = 0

    for file in os.listdir(dir_path):
        if file.startswith("bp_itr"):
            num_frame += 1
        #     bp_plot = plt.imread(dir_path + os.sep + file)
        #     all_bp_image.append(bp_plot)
        # elif file.startswith("profile_itr"):
        #     profile_plot = plt.imread(dir_path + os.sep + file)
        #     all_profile_image.append(profile_plot)

    gif_list = []

    for i in range(num_frame):
        # bp_plot = all_bp_image[i]
        # profile_plot = all_profile_image[i]

        bp_plot = plt.imread(dir_path + os.sep + "bp_itr_{}.png".format(i))
        profile_plot = plt.imread(dir_path + os.sep + "profile_itr_{}.png".format(i))

        new_img = np.concatenate([bp_plot, profile_plot], axis=1)
        plt.imsave(dir_path + os.sep + 'itr_{}.png'.format(i), new_img, dpi=300)
        new_img_uint8 = (new_img*255).astype(np.uint8)
        gif_list.append(new_img_uint8)

    imageio.mimsave(dir_path + os.sep + "rl_update.gif", gif_list, duration=1)


def plot_training_curve(file_path):
    data = pd.read_csv(file_path)
    episode = data['Episode'].values[1:]
    max_error = data['Max Error'].values[1:]
    itr = data['Iteration'].values[1:]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(episode, max_error, 'r-o', label='Average Max Execution Error')
    ax2.plot(episode, itr, 'b-o', label='Average Num. Iteration')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Max Error', color='r')
    ax2.set_ylabel('Iteration', color='b')
    fig.legend()
    ax1.set_title('RL Training Curve')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 12)

    fig.savefig('RL Train.png', dpi=300)


def max_gradient_gif():
    frame_list = []

    for i in range(50):
        frame = plt.imread("recorded_data/iteration_ {}.png".format(i))
        frame_list.append(frame)

    imageio.mimsave('recorded_data/max_gradient.gif', frame_list, duration=0.5)


def plot_q_value():
    df = pd.read_csv('train_q_values.csv')
    # df = df.loc[df['step'] < 150]
    steps = df['step'].values
    q_optimal = df['optimal'].values
    q_extreme = df['extreme'].values
    q_policy = df['policy'].values
    q_highest = df['best'].values

    best_x = df['x'].values
    best_y = df['y'].values
    best_z = df['z'].values

    fig, ax = plt.subplots()
    ax.plot(steps, q_optimal, label='optimal')
    ax.plot(steps, q_extreme, label='extreme')
    ax.plot(steps, q_policy, label='policy')
    ax.legend()
    # ax.set_yscale('log')
    ax.set_ylim(-25, 100)
    ax.set_title('Q values')
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Q Value')
    fig.savefig('Train Q Values.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(steps, (q_optimal - q_policy), label='$\Delta Q_{optimal}$')
    ax.plot(steps, (q_extreme - q_policy), label='$\Delta Q_{extreme}$')
    ax.plot(steps, (q_highest - q_policy), label='$\Delta Q_{highest}$')
    ax.plot(steps, np.zeros(len(steps)), label='Zero')
    ax.legend()
    ax.set_title('Q value diff')
    # ax.set_ylim(-50, 50)
    ax.set_xlabel('Train Step')
    ax.set_ylabel('$\Delta Q$')
    fig.savefig('Train Q Values diff.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(steps, (q_optimal - q_extreme), label='$Q_{optimal} - Q_{extreme}$')
    ax.plot(steps, (q_optimal - q_highest), label='$Q_{optimal} - Q_{highest}$')
    ax.plot(steps, np.zeros(len(steps)), label='Zero')
    ax.legend()
    # ax.set_ylim(-50, 50)
    ax.set_title('Q value diff')
    ax.set_xlabel('Train Step')
    ax.set_ylabel('$\Delta Q$')
    fig.savefig('Train Q Values diff New.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(steps, best_x, 'v', label='$a_x$')
    ax.plot(steps, best_y, 'x', label='$a_y$')
    ax.plot(steps, best_z, '.', label='$a_z$')
    ax.legend()
    ax.set_title('Highest Q Value Action')
    # ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Action Value')
    fig.savefig('Train Action Values.png', dpi=300)


if __name__ == '__main__':
    plot_q_value()
    # generate_gif("render/curve1/curve11")
    # max_gradient_gif()

    # num_curve = 1
    # for i in range(num_curve):
    #     generate_gif("render/curve2/curve{}".format(i))
    #     print("{} / {}".format(i, num_curve))

    # plot_training_curve('Eval Result.csv')

