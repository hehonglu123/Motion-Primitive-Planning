import os.path
import sys, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
# from toolbox.MotionSend import MotionSend, calc_lam_cs
# from toolbox.toolbox_circular_fit import arc_from_3point, circle_from_3point
# from toolbox.error_check import calc_all_error_w_normal, get_angle
# from toolbox.utils import car2js

sys.path.append('../toolbox/')

from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *

from general_robotics_toolbox import *
from abb_motion_program_exec_client import *
from curve_normalization import PCA_normalization

### fanuc robot
from fanuc_motion_program_exec_client import *
sys.path.append('../simulation/roboguide/fanuc_toolbox/')
from fanuc_utils import *


class ILCEnv(object):
    def __init__(self, curve, curve_R, curve_js, robot, n, mode='robot_studio'):
        self.robot = robot
        self.curve = curve
        self.curve_R = curve_R
        self.curve_js = curve_js
        self.curve_exe = None
        self.n = n
        self.action_dim = 3

        self.v = 250
        self.s = speeddata(self.v, 9999999, 9999999, 999999)
        self.z = z10

        #### fanuc speed and zone (corner path) profile
        if mode =='roboguide' or mode =='fanuc':
            self.s = self.v # mm/sec
            self.z = 100 # CNT100
        ####

        self.breakpoints = []
        self.p_bp = []
        self.next_p_bp = []
        self.q_bp = []
        self.next_q_bp = []
        self.ori_bp = []
        self.primitives = ['movel_fit'] * n

        self.bp_features = []
        self.bp_errors = []

        self.max_exec_error = 999
        self.state_curve_error = []
        self.state_curve_target = []
        self.state_robot = []
        self.state_error = []
        self.state_is_start = np.zeros(self.n)
        self.state_is_end = np.zeros(self.n)
        self.state_is_start[0] = 1
        self.state_is_end[-1] = 1
        self.itr = 0
        self.max_itr = 10

        self.reward_error_gain = -100
        self.fail_reward = -1000
        self.success_reward = 100
        self.step_reward = -1
        self.fail_error = 5
        self.success_error = 0.25
        self.success_decay_factor = 0.9

        self.exe_profile = {'Error': None, 'Angle Error': None, 'Speed': None, 'lambda': None}

        self.execution_mode = mode
        self.execution_method = {'robot_studio': self.execute_robot_studio,
                                 'abb': self.execute_abb_robot,
                                 'roboguide': self.execute_roboguide,
                                 'fanuc': self.execute_fanuc_robot}

    def initialize_breakpoints(self):
        self.breakpoints = np.linspace(1, len(self.curve), self.n).astype(int) - 1
        self.p_bp, self.q_bp = [], []
        for i in self.breakpoints:
            self.p_bp.append([self.curve[i]])
            self.q_bp.append([self.curve_js[i]])
            self.ori_bp.append([self.curve_R[i]])
        self.extend(init=True)
        self.next_p_bp = self.p_bp
        self.next_q_bp = self.q_bp

    def reward(self, curve_error):
        max_errors = np.zeros(len(curve_error))
        for i, interval_error in enumerate(curve_error):
            max_errors[i] = (np.max(np.linalg.norm(interval_error, axis=1)))

        fail = self.max_exec_error >= self.fail_error
        success = self.max_exec_error <= self.success_error
        done = success or fail or self.itr >= self.max_itr

        reward = self.reward_error_gain * max_errors + fail * self.fail_reward + success * self.success_reward * self.success_decay_factor**self.itr
        done = np.ones(len(curve_error)) if done else np.zeros(len(curve_error))

        message = "[Max error {:.4f}.]".format(self.max_exec_error)
        if success:
            message = "[SUCCESS. Max error {:.4f}.]".format(self.max_exec_error)
        elif fail:
            message = "[Fail. Max error {:.4f}.]".format(self.max_exec_error)

        return reward, done, message

    def reset(self):
        try:
            self.initialize_breakpoints()
            self.itr = 0
            error, angle_error, curve_exe, curve_exe_R, curve_target, curve_target_R = self.execution_method[self.execution_mode]()
        except:
            # traceback.print_exc()
            print("[Reset] Initialization Fail. Skipped Curve.")
            return None, False
        else:
            self.curve_exe = curve_exe
            self.state_curve_error, self.state_curve_target, self.state_robot, self.state_error = self.get_state(error, curve_target)
            state = (self.state_curve_error, self.state_curve_target, self.state_robot, self.state_is_start, self.state_is_end)
            return state, True

    def step(self, actions):
        self.itr += 1
        for i in range(1, len(self.q_bp) - 1):
            self.step_breakpoint(i, actions[i-1])

        try:
            self.next_q_bp = self.get_q_bp(self.next_p_bp)
        except:
            # traceback.print_exc()
            print("[Fail. Unreachable joint position.]")
            next_state = (np.zeros((self.n, 50, 3)), np.zeros((self.n, 50, 3)), np.zeros((self.n, 2)), self.state_is_start, self.state_is_end)
            return next_state, self.fail_reward, True, "Error"
        else:
            self.p_bp = self.next_p_bp
            self.q_bp = self.next_q_bp

            # self.extend(init=False)
            self.update_ori_bp()

            try:
                error, angle_error, curve_exe, curve_exe_R, curve_target, curve_target_R = self.execution_method[self.execution_mode]()
            except:
                traceback.print_exc()
                print("[Fail. RobotSudio Execution Error.]")
                next_state = (
                np.zeros((self.n, 50, 3)), np.zeros((self.n, 50, 3)), np.zeros((self.n, 2)), self.state_is_start,
                self.state_is_end)
                return next_state, self.fail_reward, True, "Error"
            else:
                self.curve_exe = curve_exe
                next_state_curve_error, next_state_curve_target, next_state_robot, next_state_error = self.get_state(error, curve_target)

                reward, done, message = self.reward(next_state_curve_error)

                self.state_curve_error, self.state_curve_target, self.state_robot, self.state_error = next_state_curve_error, next_state_curve_target, next_state_robot, next_state_error

                next_state = (next_state_curve_error, next_state_curve_target, next_state_robot, self.state_is_start, self.state_is_end)

                return next_state, reward, done, message

    def render(self, idx, save=False, render_profile=True, save_dir=''):
        save_dir = 'render' if save_dir == "" else save_dir
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot3D(self.curve[:, 0], self.curve[:, 1], self.curve[:, 2], 'g', label='Target')

        if self.curve_exe is not None:
            ax.plot3D(self.curve_exe[:, 0], self.curve_exe[:, 1], self.curve_exe[:, 2], 'r', label='Execution')

        breakpoints = np.zeros((self.n, 3))
        for i in range(1, len(self.p_bp) - 1):
            breakpoints[i-1, :] = self.p_bp[i][-1]
        ax.plot3D(breakpoints[:, 0], breakpoints[:, 1], breakpoints[:, 2], 'b.', label='Breakpoints')
        ax.set_xlim(1000, 1150)
        ax.set_ylim(-500, 500)
        ax.set_zlim(980, 1000)

        ax.legend()
        ax.set_title("Iteration {} (Max Error {:.3f})".format(self.itr, self.max_exec_error))

        if save:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            if not os.path.isdir('{}/curve{}'.format(save_dir, idx)):
                os.mkdir('{}/curve{}'.format(save_dir, idx))
            fig.savefig('{}/curve{}/bp_itr_{}.png'.format(save_dir, idx, self.itr), dpi=300)
        plt.close()

        error=self.exe_profile['Error']
        speed=self.exe_profile['Speed']
        ave_speed=np.mean(speed)
        angle_error=self.exe_profile['Angle Error']
        print('Max Error:',max(error),'Ave. Speed:',ave_speed,'Std. Speed:',np.std(speed),'Std/Ave (%):',np.std(speed)/ave_speed*100)
        print('Max Speed:',max(speed),'Min Speed:',np.min(speed),'Ave. Error:',np.mean(error),'Min Error:',np.min(error),"Std. Error:",np.std(error))
        print('Max Ang Error:',max(np.degrees(angle_error)),'Min Ang Error:',np.min(np.degrees(angle_error)),'Ave. Ang Error:',np.mean(np.degrees(angle_error)),"Std. Ang Error:",np.std(np.degrees(angle_error)))
        print("===========================================")

        if render_profile:
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            ax.plot(self.exe_profile['lambda'], self.exe_profile['Speed'], 'g-', label='Speed')
            ax2.plot(self.exe_profile['lambda'], self.exe_profile['Error'], 'b-', label='Error')
            ax2.plot(self.exe_profile['lambda'], np.degrees(self.exe_profile['Angle Error']), 'y-', label='Error')
            ax.set_xlabel('lambda (mm)')
            ax.set_ylabel('Speed (mm/s)', color='g')
            ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
            ax.set_ylim(0, 1.2*self.v)
            ax2.set_ylim(0, 2)
            ax.set_title("[Error ({:.3f},{:.3f})] [Speed ({:.2f},{:.2f})]".format(np.mean(self.exe_profile['Error']),
                                                                                  np.std(self.exe_profile['Error']),
                                                                                  np.mean(self.exe_profile['Speed']),
                                                                                  np.std(self.exe_profile['Speed'])))

            if save:
                fig.savefig('{}/curve{}/profile_itr_{}.png'.format(save_dir, idx, self.itr), dpi=300)

            plt.close()

    def step_breakpoint(self, bp_idx, action):
        cur_bp = self.p_bp[bp_idx][-1]
        prev_bp = self.p_bp[bp_idx - 1][-1]
        next_bp = self.p_bp[bp_idx + 1][-1]
        error_val = np.linalg.norm(self.state_error[bp_idx-1])

        dir0 = self.state_error[bp_idx-1] / error_val
        dir1 = (prev_bp - cur_bp) / np.linalg.norm(prev_bp - cur_bp)
        dir2 = (next_bp - cur_bp) / np.linalg.norm(next_bp - cur_bp)

        new_p_bp = cur_bp + error_val * (action[0] * dir0 + action[1] * dir1 + action[2] * dir2)
        self.next_p_bp[bp_idx] = [new_p_bp]

    def update_ori_bp(self):
        for i in range(1, len(self.ori_bp)-1):
            point_dist = np.linalg.norm(self.curve - self.p_bp[i][-1], axis=-1)
            point_idx = np.argmin(point_dist)
            self.ori_bp[i] = [self.curve_R[point_idx]]

    def get_q_bp(self, p_bp):
        q_bp = []
        for i in range(len(p_bp)):
            q_bp.append(car2js(self.robot, self.q_bp[i][-1], p_bp[i][-1], self.robot.fwd(self.q_bp[i][-1]).R))
        return q_bp

    def get_state(self, error, curve_target):
        state_curve_error = []
        state_curve_target = []
        state_robot = []
        state_error = []

        if len(error) != len(curve_target):
            raise Exception("ERROR in env.get_state(): error curve and target curve do not have the same length.")

        bp_info = []
        for i in range(1, len(self.p_bp) - 1):
            p_bp = self.p_bp[i][-1]
            closest_point_idx = np.argmin(np.linalg.norm(curve_target - p_bp, axis=1))
            error_dir = error[closest_point_idx]
            bp_info.append([i, closest_point_idx, error_dir])
            state_error.append(error_dir)

        for i in range(len(bp_info)):
            prev_point_idx = 0 if i == 0 else bp_info[i-1][1]
            next_point_idx = len(error) if i == len(bp_info) - 1 else bp_info[i+1][1]
            next_point_idx = max(prev_point_idx+1, next_point_idx)

            local_error_traj = error[prev_point_idx:next_point_idx]
            local_target_traj = curve_target[prev_point_idx:next_point_idx]
            local_error_traj = PCA_normalization(local_error_traj, n_points=50, rescale=False)
            local_target_traj = PCA_normalization(local_target_traj, n_points=50, rescale=True)

            # state_curve.append(np.hstack([local_error_traj.flatten(), local_target_traj.flatten()]))
            state_curve_error.append(local_error_traj)
            state_curve_target.append(local_target_traj)

        for i in range(1, len(self.p_bp) - 1):
            prev_robot_feature = 0
            if i > 1:
                prev_q_bp = self.q_bp[i-1][-1]
                jac_inv = np.linalg.pinv(self.robot.jacobian(prev_q_bp))
                p_dist = self.p_bp[i][-1] - self.p_bp[i-1][-1]
                p_ori_diff = self.ori_bp[i-1][-1] - self.ori_bp[i-1-1][-1]
                p_diff = np.concatenate([p_dist, p_ori_diff], axis=-1)
                prev_robot_feature = np.linalg.norm(np.matmul(jac_inv, p_diff))

            next_robot_feature = 0
            if i < len(self.p_bp) - 2:
                jac_inv = np.linalg.pinv(self.robot.jacobian(self.q_bp[i][-1]))
                p_dist = self.p_bp[i+1][-1] - self.p_bp[i][-1]
                p_ori_diff = self.ori_bp[i+1-1][-1] - self.ori_bp[i-1][-1]
                p_diff = np.concatenate([p_dist, p_ori_diff], axis=-1)
                next_robot_feature = np.linalg.norm(np.matmul(jac_inv, p_diff))
            state_robot.append([prev_robot_feature, next_robot_feature])

        return state_curve_error, state_curve_target, state_robot, state_error

    def get_error_speed(self, error, angle_error, speed, lam):
        error_profile = np.linalg.norm(error, axis=-1)

        self.max_exec_error = np.max(error_profile)
        self.exe_profile['Error'] = error_profile
        self.exe_profile['Angle Error'] = angle_error
        self.exe_profile['Speed'] = speed
        self.exe_profile['lambda'] = lam

        return self.exe_profile

    def execute_robot_studio(self):
        ms = MotionSend()
        primitives = self.primitives.copy()
        primitives.insert(0, 'movej_fit')
        primitives.append('movel_fit')

        breakpoints = np.hstack([-1, self.breakpoints, -1])
        ms.write_data_to_cmd('recorded_data/command.csv', breakpoints, primitives, self.p_bp, self.q_bp)

        logged_data = ms.exec_motions(self.robot, primitives, self.breakpoints, self.p_bp, self.q_bp, self.s, self.z)
        with open("recorded_data/curve_exe_v" + str(self.v) + "_z10.csv", "w") as f:
            f.write(logged_data)

        StringData = StringIO(logged_data)
        df = pd.read_csv(StringData, sep=",")

        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = ms.logged_data_analysis(self.robot, df, realrobot=False)
        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = self.chop_extension(curve_exe, curve_exe_R, curve_exe_js, speed, timestamp)
        curve_exe, curve_exe_R, curve_target, curve_target_R = self.interpolate_curve(curve_exe, curve_exe_R)
        error, angle_error = self.calculate_error(curve_exe, curve_exe_R, curve_target, curve_target_R)
        self.get_error_speed(error, angle_error, speed, lam)
        return error, angle_error, curve_exe, curve_exe_R, curve_target, curve_target_R

    def execute_abb_robot(self):
        ms = MotionSend(url='http://192.168.55.1:80')
        primitives = self.primitives.copy()
        primitives.insert(0, 'movej_fit')
        primitives.append('movel_fit')

        breakpoints = np.hstack([-1, self.breakpoints, -1])
        ms.write_data_to_cmd('recorded_data/command.csv', breakpoints, primitives, self.p_bp, self.q_bp)

        ###5 run execute
        curve_exe_all=[]
        curve_exe_js_all=[]
        timestamp_all=[]
        total_time_all=[]

        for r in range(5):

            logged_data=ms.exec_motions(self.robot,primitives,self.breakpoints,self.p_bp,self.q_bp,self.s,self.z)

            StringData=StringIO(logged_data)
            df = read_csv(StringData, sep =",")
            ##############################data analysis#####################################
            lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(self.robot,df,realrobot=True)

            fig, ax1 = plt.subplots()
            ax1.plot(lam[1:], speed, 'g-', label='Speed')
            ax1.axis(ymin=0,ymax=1.2*self.v)

            ax1.set_xlabel('lambda (mm)')
            ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
            plt.title("Speed and Error Plot")
            ax1.legend(loc=0)

            plt.legend()
            plt.savefig('recorded_data/'+str(self.itr)+'_run_'+str(r))
            plt.clf()
            ###throw bad curves
            _, _, _,_, _, timestamp_temp=self.chop_extension(curve_exe, curve_exe_R, curve_exe_js, speed, timestamp)
            total_time_all.append(timestamp_temp[-1]-timestamp_temp[0])

            timestamp=timestamp-timestamp[0]

            curve_exe_all.append(curve_exe)
            curve_exe_js_all.append(curve_exe_js)
            timestamp_all.append(timestamp)

        ###trajectory outlier detection, based on chopped time
        curve_exe_all,curve_exe_js_all,timestamp_all=remove_traj_outlier(curve_exe_all,curve_exe_js_all,timestamp_all,total_time_all)

        ###infer average curve from linear interplateion
        curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)


        ###calculat data with average curve
        lam, curve_exe, curve_exe_R, speed=logged_data_analysis(self.robot,timestamp_d,avg_curve_js)

        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = self.chop_extension(curve_exe, curve_exe_R, curve_exe_js, speed, timestamp_d)
        curve_exe, curve_exe_R, curve_target, curve_target_R = self.interpolate_curve(curve_exe, curve_exe_R)
        error, angle_error = self.calculate_error(curve_exe, curve_exe_R, curve_target, curve_target_R)
        self.get_error_speed(error, angle_error, speed, lam)
        return error, angle_error, curve_exe, curve_exe_R, curve_target, curve_target_R

    def execute_roboguide(self):

        ms = MotionSendFANUC()
        primitives = self.primitives.copy()
        primitives.insert(0, 'movej_fit')
        primitives.append('movel_fit')

        breakpoints = np.hstack([-1, self.breakpoints, -1])
        ms.write_data_to_cmd(os.getcwd()+'/recorded_data/command.csv', breakpoints, primitives, self.p_bp, self.q_bp)

        logged_data = ms.exec_motions(self.robot, primitives, self.breakpoints, self.p_bp, self.q_bp, self.s, self.z)
        with open(os.getcwd()+"/recorded_data/curve_exe_v" + str(self.v) + "_CNT"+str(self.z)+".csv", "wb") as f:
            f.write(logged_data)

        StringData = StringIO(logged_data.decode('utf-8'))
        df = pd.read_csv(StringData)

        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = ms.logged_data_analysis(self.robot, df)
        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = self.chop_extension(curve_exe, curve_exe_R, curve_exe_js, speed, timestamp)
        curve_exe, curve_exe_R, curve_target, curve_target_R = self.interpolate_curve(curve_exe, curve_exe_R)
        error, angle_error = self.calculate_error(curve_exe, curve_exe_R, curve_target, curve_target_R)
        self.get_error_speed(error, angle_error, speed, lam)
        return error, angle_error, curve_exe, curve_exe_R, curve_target, curve_target_R

    def execute_fanuc_robot(self):
        
        raise Exception("execute_fanuc_robot() not implemented!")

    def calculate_error(self, curve_exe, curve_exe_R, curve_target, curve_target_R):
        error = curve_target - curve_exe
        angle_error = np.zeros(len(curve_exe))
        for i in range(len(curve_exe)):
            normal_angle = get_angle(curve_exe_R[i], curve_target_R[i])
            angle_error[i] = normal_angle
        return error, angle_error

    def interpolate_curve(self, curve_exe, curve_exe_R, keep_target_length=False):
        if keep_target_length:
            new_curve_exe = np.zeors((len(curve_exe), 3))
            new_curve_exe_R = np.zeros((len(curve_exe), 3))
            exec_point_idx = []
            for i in range(len(curve_exe)):
                dist = np.linalg.norm(self.curve - curve_exe[i], axis=1)
                closest_point_idx = np.argmin(dist)
                exec_point_idx.append((closest_point_idx, curve_exe[i, :], curve_exe_R[i, :, -1]))

            if exec_point_idx[0][0] > 0:
                exec_point_idx.insert(0, (0, self.curve[0]))
            if exec_point_idx[-1][0] < len(self.curve) - 1:
                exec_point_idx.append((len(self.curve) - 1, self.curve[-1]))

            for i in range(len(exec_point_idx) - 1):
                start_idx, start_point, start_point_R = exec_point_idx[i]
                end_idx, end_point, end_point_R = exec_point_idx[i + 1]
                num_points = end_idx - start_idx
                n = np.linspace(0, num_points, num_points + 1)
                slope = (end_point - start_point) / num_points
                fit = n * slope + start_point
                new_curve_exe[start_idx:end_idx + 1, :] = fit

                slope_R = (end_point_R - start_point_R) / num_points
                fit_R = n * slope_R + start_point_R
                new_curve_exe_R[start_idx:end_idx + 1, :, :] = fit_R

            return new_curve_exe, new_curve_exe_R, self.curve, self.curve_R
        else:
            new_curve_target = np.zeros((len(curve_exe), 3))
            new_curve_target_R = np.zeros((len(curve_exe), 3))
            for i in range(len(curve_exe)):
                dist = np.linalg.norm(self.curve - curve_exe[i], axis=1)
                closest_point_idx = np.argmin(dist)
                new_curve_target[i, :] = self.curve[closest_point_idx, :]
                new_curve_target_R[i, :] = self.curve_R[closest_point_idx, :]
            return curve_exe, curve_exe_R[:, :, -1], new_curve_target, new_curve_target_R

    def extend(self, extension_start=50, extension_end=50, init=False):
        # initial point extension
        if init:
            pose_start = self.robot.fwd(self.q_bp[0][-1])
            p_start = pose_start.p
            R_start = pose_start.R
            pose_end = self.robot.fwd(self.q_bp[1][-1])
            p_end = pose_end.p
            R_end = pose_end.R
        else:
            pose_start = self.robot.fwd(self.q_bp[1][-1])
            p_start = pose_start.p
            R_start = pose_start.R
            pose_end = self.robot.fwd(self.q_bp[2][-1])
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
        if init:
            q_start_new = car2js(self.robot, self.q_bp[0][0], p_start_new, R_start_new)[0]
        else:
            q_start_new = car2js(self.robot, self.q_bp[1][0], p_start_new, R_start_new)[0]

        # end point extension
        if init:
            pose_start = self.robot.fwd(self.q_bp[-2][-1])
            p_start = pose_start.p
            R_start = pose_start.R
            pose_end = self.robot.fwd(self.q_bp[-1][-1])
            p_end = pose_end.p
            R_end = pose_end.R
        else:
            pose_start = self.robot.fwd(self.q_bp[-3][-1])
            p_start = pose_start.p
            R_start = pose_start.R
            pose_end = self.robot.fwd(self.q_bp[-2][-1])
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
        if init:
            q_end_new = car2js(self.robot, self.q_bp[-1][0], p_end_new, R_end_new)[0]
        else:
            q_end_new = car2js(self.robot, self.q_bp[-2][0], p_end_new, R_end_new)[0]

        if init:
            self.p_bp.insert(0, [p_start_new])
            self.p_bp.append([p_end_new])
            self.q_bp.insert(0, [q_start_new])
            self.q_bp.append([q_end_new])
        else:
            self.p_bp[0] = [p_start_new]
            self.p_bp[-1] = [p_end_new]
            self.q_bp[0] = [q_start_new]
            self.q_bp[-1] = [q_end_new]

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
