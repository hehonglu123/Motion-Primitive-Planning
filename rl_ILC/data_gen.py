import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import sys
from scipy.optimize import fminbound
sys.path.append('../toolbox')
from lambda_calc import *
from utils import *
sys.path.append('../data')
from baseline import pose_opt, curve_frame_conversion, find_js


def curve3_points(num_points=5000):
    lam = np.linspace(0, 1000, num_points)
    d_lam = lam[1] - lam[0]

    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)

    x[lam<=250] = 0
    x[(250<lam) * (lam<750)] = np.linspace(0, 500-d_lam, sum((250<lam) * (lam<750)))
    x[lam>=750] = 500

    y[lam<=250] = np.linspace(0, 250-d_lam, sum(lam<=250))
    y[(250<lam) * (lam<750)] = 250
    y[lam>=750] = np.linspace(250-d_lam, 0, sum(lam>=750))

    xn = np.zeros(num_points)
    yn = np.zeros(num_points)
    zn = -np.ones(num_points)

    curve = np.vstack([y, x, z]).T
    curve_normal = np.vstack([yn, xn, zn]).T

    return curve, curve_normal


def curve4_points(num_points=5000):
    lam = np.linspace(0, 1000, num_points)
    d_lam = lam[1] - lam[0]

    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)

    x[lam <= 500] = 0
    x[lam >= 500] = np.linspace(d_lam, 500/2, sum(lam >= 500))

    y[lam <= 500] = np.linspace(0, 500, sum(lam <= 500))
    y[lam >= 500] = np.linspace(500 + d_lam, 500+500*np.sqrt(3)/2, sum(lam >= 500))

    xn = np.zeros(num_points)
    yn = np.zeros(num_points)
    zn = -np.ones(num_points)

    curve = np.vstack([x, y, z]).T
    curve_normal = np.vstack([xn, yn, zn]).T

    return curve, curve_normal


def main():

    robot = abb6640(d=50)
    curve, curve_normal = curve4_points()

    # offset = [800, -250, 800]
    # curve = curve + offset

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot3D(curve[:, 0], curve[:, 1], curve[:, 2], 'r.-')
    ax.quiver(curve[::100, 0], curve[::100, 1], curve[::100, 2], curve_normal[::100, 0], curve_normal[::100, 1],
              curve_normal[::100, 2], length=5, normalize=True)
    ax.plot3D(curve[0, 0], curve[0, 1], curve[0, 2], 'rX')
    ax.plot3D(curve[-1, 0], curve[-1, 1], curve[-1, 2], 'bX')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    plt.show()

    # curve_base, curve_normal_base = curve, curve_normal
    #
    # print(curve[1245:1255])
    # print(curve_normal[1245:1255])
    #
    # curve_js_all = find_js(robot, curve_base, curve_normal_base)
    # J_min = []
    # for i in range(len(curve_js_all)):
    #     J_min.append(find_j_min(robot, curve_js_all[i]))
    #
    # J_min = np.array(J_min)
    # print(J_min)
    # curve_js = curve_js_all[np.argmin(J_min.min(axis=1))]
    #
    # # if show:
    # #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # #     ax.plot3D(curve_base[:, 0], curve_base[:, 1], curve_base[:, 2], 'r.-')
    # #     plt.show()
    #
    # df_base = DataFrame(
    #     {'x': curve_base[:, 0], 'y': curve_base[:, 1], 'z': curve_base[:, 2], 'x_dir': curve_normal_base[:, 0],
    #      'y_dir': curve_normal_base[:, 1], 'z_dir': curve_normal_base[:, 2]})
    # df_base.to_csv(save_dir + os.sep + 'base/curve_{}.csv'.format(curve_idx), header=False, index=False)
    #
    # df_js = DataFrame(curve_js)
    # df_js.to_csv(save_dir + os.sep + 'js/curve_{}.csv'.format(curve_idx), header=False, index=False)

if __name__ == '__main__':
    main()

