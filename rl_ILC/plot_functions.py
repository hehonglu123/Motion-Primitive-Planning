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
    num_file = len(os.listdir(dir_path))

    all_bp_image = []
    all_profile_image = []

    for file in os.listdir(dir_path):
        if file.startswith("bp_itr"):
            bp_plot = plt.imread(dir_path + os.sep + file)
            all_bp_image.append(bp_plot)
        elif file.startswith("profile_itr"):
            profile_plot = plt.imread(dir_path + os.sep + file)
            all_profile_image.append(profile_plot)

    gif_list = []

    for i in range(len(all_bp_image)):
        bp_plot = all_bp_image[i]
        profile_plot = all_profile_image[i]

        new_img = np.concatenate([bp_plot, profile_plot], axis=1)
        plt.imsave(dir_path + os.sep + 'itr_{}.png'.format(i), new_img, dpi=300)
        new_img_uint8 = (new_img*255).astype(np.uint8)
        gif_list.append(new_img_uint8)

    imageio.mimsave(dir_path + os.sep + "rl_update.gif", gif_list, duration=1)


if __name__ == '__main__':
    generate_gif("render/curve1")

