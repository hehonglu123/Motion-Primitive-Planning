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
    episode = data['Episode'].values
    max_error = data['Max Error'].values
    itr = data['Iteration'].values

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(episode, max_error, 'r-o', label='Average Max Execution Error')
    ax2.plot(episode, itr, 'b-o', label='Average Num. Iteration')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Max Error', color='r')
    ax2.set_ylabel('Iteration', color='b')
    fig.legend()
    ax1.set_title('RL Training Curve')

    fig.savefig('RL Train.png', dpi=300)


def max_gradient_gif():
    frame_list = []

    for i in range(50):
        frame = plt.imread("recorded_data/iteration_ {}.png".format(i))
        frame_list.append(frame)

    imageio.mimsave('recorded_data/max_gradient.gif', frame_list, duration=0.5)


if __name__ == '__main__':
    # generate_gif("render/curve1/curve10")
    # max_gradient_gif()

    # num_curve = 1
    # for i in range(num_curve):
    #     generate_gif("render/curve2/curve{}".format(i))
    #     print("{} / {}".format(i, num_curve))

    plot_training_curve('Eval Result New.csv')

