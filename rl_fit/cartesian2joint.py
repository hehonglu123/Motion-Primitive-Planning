import numpy as np
import pandas as pd
from pandas import *
import sys, traceback
import os
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../toolbox')
from robots_def import *
from utils import *


def main():
    data_dir = "data/base"
    abb6640_obj=abb6640(d=50)

    col_names = ['X', 'Y', 'Z', 'direction_x', 'direction_y', 'direction_z']
    for file in os.listdir(data_dir):
        print("Trajectory: {}".format(file))

        file_name = os.path.splitext(file)[0]
        file_path = data_dir + os.sep + file
        data = pd.read_csv(file_path, names=col_names)
        curve_x = data['X'].tolist()
        curve_y = data['Y'].tolist()
        curve_z = data['Z'].tolist()
        curve_direction_x = data['direction_x'].tolist()
        curve_direction_y = data['direction_y'].tolist()
        curve_direction_z = data['direction_z'].tolist()

        curve = np.vstack((curve_x, curve_y, curve_z)).T
        curve_direction = np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

        curve_R = []

        for i in range(len(curve) - 1):
            try:
                R_curve = direction2R(curve_direction[i], -curve[i + 1] + curve[i])
                if i > 0:
                    k, angle_of_change = R2rot(np.dot(curve_R[-1], R_curve.T))
                    if angle_of_change > 0.1:
                        curve_R.append(curve_R[-1])
                        continue
            except:
                traceback.print_exc()
                pass

            curve_R.append(R_curve)

        # insert initial orientation
        curve_R.insert(0, curve_R[0])
        curve_js = np.zeros((len(curve), 6))

        # q_init=np.radians([35.414132, 12.483655, 27.914093, -89.255298, 51.405928, -128.026891])
        q_init = np.array([0.625835928, 0.836930134, -0.239948016, 1.697010866, -0.89108048, 0.800838687])

        no_solution = False
        for i in range(len(curve)):
            if no_solution:
                break
            try:
                q_all = np.array(abb6640_obj.inv(curve[i], curve_R[i]))
            except:
                traceback.print_exc()
                pass
            # choose inv_kin closest to previous joints
            if i == 0:
                try:
                    temp_q = q_all - q_init
                    order = np.argsort(np.linalg.norm(temp_q, axis=1))
                    curve_js[i] = q_all[order[0]]
                except:
                    traceback.print_exc()
                    no_solution = True
                    break

            else:
                try:
                    temp_q = q_all - curve_js[i - 1]
                    order = np.argsort(np.linalg.norm(temp_q, axis=1))
                    curve_js[i] = q_all[order[0]]

                except:
                    q_all = np.array(abb6640_obj.inv(curve[i], curve_R[i]))
                    traceback.print_exc()
                    pass

        ###output to csv
        if not no_solution:
            df = DataFrame({'q0': curve_js[:, 0], 'q1': curve_js[:, 1], 'q2': curve_js[:, 2], 'q3': curve_js[:, 3],
                            'q4': curve_js[:, 4], 'q5': curve_js[:, 5]})
            df.to_csv('data/js_new/{}_js_new.csv'.format(file_name), header=False, index=False)


if __name__ == "__main__":
    main()
