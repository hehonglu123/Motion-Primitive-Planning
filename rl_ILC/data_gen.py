import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from scipy.optimize import fminbound
sys.path.append('../toolbox')
from lambda_calc import *
from utils import *
from data.baseline import pose_opt, curve_frame_conversion, find_js


def curve3_points(num_points=50000):
    lam = np.linspace(0, 1000, num_points)

    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)

    x[lam<=250] = 0
    x[(250<lam) * (lam<750)] = np.linspace(1, 500, sum((250<lam) * (lam<750)))
    x[lam>=750] = 500

    y[lam<=250] = np.linspace(0, 250, sum(lam<=250))
    y[(250<lam) * (lam<750)] = 250
    y[lam>=750] = np.linspace(249, 0, sum(lam>=750))

    xn = np.zeros(num_points)
    yn = np.zeros(num_points)
    zn = -np.ones(num_points)

    curve = np.vstack([x, y, z]).T
    curve_normal = np.vstack([xn, yn, zn]).T

    return curve, curve_normal


def curve4_points():
    x = np.linspace(0, 1000, 1000)
    y = np.zeros(1000)
    z = np.zeros(1000)

    y[500:] = np.linspace(0, 500/np.sqrt(3), 500)

    xn = np.zeros(1000)
    yn = np.zeros(1000)
    zn = -np.ones(1000)

    curve = np.vstack([x, y, z]).T
    curve_normal = np.vstack([xn, yn, zn]).T

    return curve, curve_normal


def main():
    save_dir = 'train_data/curve3'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    robot = abb6640(d=50)
    curve, curve_normal = curve3_points()

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2], 'r.-')
    ax.quiver(curve[::1000, 0], curve[::1000, 1], curve[::1000, 2], curve_normal[::1000, 0], curve_normal[::1000, 1],
                      curve_normal[::1000, 2], length=5, normalize=True)
    ax.set_zlim(-10, 10)
    plt.show()

    H_pose = pose_opt(robot, curve, curve_normal)
    curve_base, curve_normal_base = curve_frame_conversion(curve, curve_normal, H_pose)

    curve_js_all = find_js(robot, curve_base, curve_normal_base)
    J_min = []
    for i in range(len(curve_js_all)):
        J_min.append(find_j_min(robot, curve_js_all[i]))

    J_min = np.array(J_min)
    curve_js = curve_js_all[np.argmin(J_min.min(axis=1))]

    # if show:
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     ax.plot3D(curve_base[:, 0], curve_base[:, 1], curve_base[:, 2], 'r.-')
    #     plt.show()

    df_base = DataFrame(
        {'x': curve_base[:, 0], 'y': curve_base[:, 1], 'z': curve_base[:, 2], 'x_dir': curve_normal_base[:, 0],
         'y_dir': curve_normal_base[:, 1], 'z_dir': curve_normal_base[:, 2]})
    df_base.to_csv(save_dir + os.sep + 'base/curve_{}.csv'.format(curve_idx), header=False, index=False)

    df_js = DataFrame(curve_js)
    df_js.to_csv(save_dir + os.sep + 'js/curve_{}.csv'.format(curve_idx), header=False, index=False)

if __name__ == '__main__':
    main()

