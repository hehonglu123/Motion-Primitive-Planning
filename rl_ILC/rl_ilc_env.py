import numpy as np
import pandas as pd

from io import StringIO
from toolbox.MotionSend import MotionSend, calc_lam_cs
from toolbox.toolbox_circular_fit import arc_from_3point, circle_from_3point
from toolbox.error_check import calc_all_error_w_normal
from toolbox.utils import car2js
from general_robotics_toolbox import *
from abb_motion_program_exec_client import *


class ILCEnv(object):
    def __init__(self, curve, curve_js, robot, n):
        self.robot = robot
        self.curve = curve
        self.curve_js = curve_js
        self.n = n

        self.v = 250
        self.s = speeddata(self.v, 9999999, 9999999, 999999)
        self.z = z10

        self.breakpoints = []
        self.p_bp = []
        self.next_p_bp = []
        self.q_bp = []
        self.next_q_bp = []
        self.primitives = ['movel_fit'] * n

        self.bp_features = []
        self.bp_errors = []

    def initialize_breakpoints(self):
        self.breakpoints = np.linspace(0, len(self.curve), self.n + 1)
        self.p_bp, self.q_bp = [], []
        for i in self.breakpoints:
            self.p_bp.append([self.curve[i]])
            self.q_bp.append([self.curve_js[i]])
        self.extend()
        self.next_p_bp = self.p_bp
        self.next_q_bp = self.q_bp

    def reward(self):
        pass

    def reset(self):
        self.initialize_breakpoints()
        ms = MotionSend()
        logged_data = ms.exec_motions(self.robot, self.primitives, )

    def step(self, actions):
        for i in range(1, len(self.q_bp) - 1):
            self.step_breakpoint(i, actions[i])

    def step_breakpoint(self, bp_idx, action):
        cur_bp = self.p_bp[bp_idx]
        prev_bp = self.p_bp[bp_idx - 1]
        next_bp = self.p_bp[bp_idx + 1]
        error_val = np.linalg.norm(self.bp_errors[bp_idx])

        dir0 = self.bp_errors[bp_idx] / error_val
        dir1 = (prev_bp - cur_bp) / np.linalg.norm(prev_bp - cur_bp)
        dir2 = (next_bp - cur_bp) / np.linalg.norm(next_bp - cur_bp)

        new_p_bp = cur_bp + error_val * (action[0] * dir0 + action[1] * dir1 + action[2] * dir2)
        self.next_p_bp[bp_idx] = new_p_bp

    def execute_robot_studio(self):
        ms = MotionSend()
        primitives = self.primitives.copy()
        primitives.insert(0, 'movej_fit')

        logged_data = ms.exec_motions(self.robot, primitives, self.breakpoints, self.p_bp, self.q_bp, self.s, self.z)
        with open("recorded_data/curve_exe_v" + str(self.v) + "_z10.csv", "w") as f:
            f.write(logged_data)
        ms.write_data_to_cmd('recorded_data/command.csv', self.breakpoints, primitives, self.p_bp, self.q_bp)
        StringData = StringIO(logged_data)
        df = pd.read_csv(StringData, sep=",")

        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = ms.logged_data_analysis(self.robot, df, realrobot=False)
        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = self.chop_extension(curve_exe, curve_exe_R, curve_exe_js, speed, timestamp)
        interp_curve_exe, interp_curve_exe_R = self.interpolate_exec_curve(curve_exe, curve_exe_R)
        error, angle_error = calc_all_error_w_normal(curve_exe, self.curve[:, :3], curve_exe_R[:, :, -1], self.curve[:, 3:])

    def interpolate_exec_curve(self, curve_exe, curve_exe_R):
        new_curve_exe = np.zeors((len(self.curve), 3))
        new_curve_exe_R = np.zeros((len(self.curve), 3, 3))
        exec_point_idx = []
        for i in range(len(curve_exe)):
            dist = np.linalg.norm(self.curve - curve_exe[i], axis=1)
            closest_point_idx = np.argmin(dist)
            exec_point_idx.append((closest_point_idx, curve_exe[i, :], curve_exe_R[i, :, -1]))

        if exec_point_idx[0][0] > 0:
            exec_point_idx.insert(0, (0, self.curve[0]))
        if exec_point_idx[-1][0] < len(self.curve) - 1:
            exec_point_idx.append((len(self.curve)-1, self.curve[-1]))

        for i in range(len(exec_point_idx)-1):
            start_idx, start_point, start_point_R = exec_point_idx[i]
            end_idx, end_point, end_point_R = exec_point_idx[i+1]
            num_points = end_idx - start_idx
            n = np.linspace(0, num_points, num_points + 1)
            slope = (end_point - start_point) / num_points
            fit = n * slope + start_point
            new_curve_exe[start_idx:end_idx + 1, :] = fit

            slope_R = (end_point_R - start_point_R) / num_points
            fit_R = n * slope_R + start_point_R
            new_curve_exe_R[start_idx:end_idx + 1, :, :] = fit_R

        return new_curve_exe, new_curve_exe_R


    def calculate_exec_error(self, curve_exe, curve_exe_R):
        exec_error_traj = np.zeros(len(self.curve), 3)
        for i in range(len(curve_exe)):
            dist = np.linalg.norm(self.curve - curve_exe[i], axis=1)
            closest_point_idx = np.argmin(dist)
            error = dist[closest_point_idx]
            error3d = self.curve[closest_point_idx] - curve_exe[i]
            exec_error_traj[i, :] = error3d

    def extend(self, extension_start=100, extension_end=100):
        # initial point extension
        pose_start = self.robot.fwd(self.q_bp[0][-1])
        p_start = pose_start.p
        R_start = pose_start.R
        pose_end = self.robot.fwd(self.q_bp[1][-1])
        p_end = pose_end.p
        R_end = pose_end.R

        # find new start point
        slope_p = p_end - p_start
        slope_p = slope_p / np.linalg.norm(slope_p)
        p_start_new = p_start - extension_start * slope_p  # extend 5cm backward

        # find new start orientation
        k, theta = R2rot(R_end @ R_start.T)
        theta_new = -extension_start * theta / np.linalg.norm(p_end - p_start)
        R_start_new = rot(k, theta_new) @ R_start

        # solve invkin for initial point
        q_start_new = car2js(self.robot, self.q_bp[0][0], p_start_new, R_start_new)[0]

        # end point extension
        pose_start = self.robot.fwd(self.q_bp[-2][-1])
        p_start = pose_start.p
        R_start = pose_start.R
        pose_end = self.robot.fwd(self.q_bp[-1][-1])
        p_end = pose_end.p
        R_end = pose_end.R

        # find new end point
        slope_p = (p_end - p_start) / np.linalg.norm(p_end - p_start)
        p_end_new = p_end + extension_end * slope_p  # extend 5cm backward

        # find new end orientation
        k, theta = R2rot(R_end @ R_start.T)
        slope_theta = theta / np.linalg.norm(p_end - p_start)
        R_end_new = rot(k, extension_end * slope_theta) @ R_end

        # solve invkin for end point
        q_end_new = car2js(self.robot, self.q_bp[-1][0], p_end_new, R_end_new)[0]

        self.p_bp.insert(0, [p_start_new])
        self.p_bp.append([p_end_new])
        self.q_bp.insert(0, [q_start_new])
        self.q_bp.append([q_end_new])

    def chop_extension(self, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp):
        p_start = self.curve[0, :3]
        p_end = self.curve[-1, :3]

        start_idx = np.argmin(np.linalg.norm(p_start-curve_exe, axis=1))
        end_idx = np.argmin(np.linalg.norm(p_end-curve_exe, axis=1))

        # make sure extension doesn't introduce error
        if np.linalg.norm(curve_exe[start_idx]-p_start) > 0.5:
            start_idx += 1
        if np.linalg.norm(curve_exe[end_idx]-p_end) > 0.5:
            end_idx -= 1

        curve_exe = curve_exe[start_idx:end_idx+1]
        curve_exe_js = curve_exe_js[start_idx:end_idx+1]
        curve_exe_R = curve_exe_R[start_idx:end_idx+1]
        speed = speed[start_idx:end_idx+1]
        lam = calc_lam_cs(curve_exe)

        return lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp[start_idx:end_idx+1]-timestamp[start_idx]
