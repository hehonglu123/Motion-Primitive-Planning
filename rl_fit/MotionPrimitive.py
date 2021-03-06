from io import StringIO

import pandas as pd

from greedy_fitting.greedy import greedy_fit
from toolbox.MotionSend import *
from rl_fit.utils.curve_normalization import PCA_normalization


class MotionPrimitiveEnv(greedy_fit):
    def __init__(self, curve_js, args):
        super(MotionPrimitiveEnv, self).__init__(args.robot, curve_js, args.max_error_threshold, args.max_ori_threshold)

        self.action_dim = args.action_dim
        self.action_key = ['movel_fit'] * args.discrete_actions + ['movej_fit'] * args.discrete_actions + ['movec_fit'] * args.discrete_actions
        self.action_length = [(i+1)/args.discrete_actions for i in range(args.discrete_actions)] * 3
        self.robot = args.robot
        self.curve_js = curve_js

        self.step_curve_fit = {}
        self.step_curve_fit_R = {}
        self.step_curve_fit_js = {}
        self.step_max_error = {}
        self.step_max_ori_error = {}
        self.step_max_error_place = {}
        self.step_slope_diff_js = {}

        self.max_error = -1
        self.max_ori_error = -1
        self.primitive_choices = []
        self.primitive_fits = []
        self.points = []
        self.q_bp = []
        self.breakpoints = [0]

        self.exec_time = -1
        self.exec_max_error = -1
        self.exec_max_normal_error= -1

        self.error_curve = None
        self.normal_error_curve = None

    def greedy_primitives_fit(self):
        for key in self.primitives:
            curve_fit, curve_fit_R, curve_fit_js, error, ori_error = self.bisect(self.primitives[key], self.breakpoints[-1], rl=True)
            if key != 'movej_fit':
                curve_fit_js = self.car2js(curve_fit, curve_fit_R)
            self.step_curve_fit[key], self.step_curve_fit_R[key], self.step_curve_fit_js[key] = curve_fit, curve_fit_R, curve_fit_js
            self.step_max_error[key] = np.max(error)
            self.step_max_ori_error[key] = np.max(ori_error)
            self.step_max_error_place[key] = np.argmax(error) / len(error)
            next_point = self.breakpoints[-1] + len(self.step_curve_fit[key])
            if 2 <= len(self.curve_fit_js) and next_point < len(self.curve) - 1:
                slope_next = (self.curve_js[next_point + 1] - self.curve_js[next_point]) / (self.lam[next_point + 1] - self.lam[next_point])
                slope = (curve_fit_js[-1] - curve_fit_js[-2]) / (self.lam[next_point] - self.lam[next_point - 1])
                self.step_slope_diff_js[key] = np.abs(slope_next - slope)
            else:
                self.step_slope_diff_js[key] = np.zeros(6)

    def reset(self):
        normalized_curve = PCA_normalization(self.curve)
        self.greedy_primitives_fit()
        primitive_features = np.array([self.step_max_error_place['movel_fit'], self.step_max_error_place['movej_fit'],
                                       self.step_max_error_place['movec_fit']])
        primitive_features = np.concatenate([primitive_features, self.step_slope_diff_js['movel_fit'], self.step_slope_diff_js['movej_fit'], self.step_slope_diff_js['movec_fit']])
        state = (normalized_curve, primitive_features)
        return state

    def rl_step(self, action):
        primitive_key = self.action_key[action]
        primitive_length = self.action_length[action]
        old_primitive_fit = self.step_curve_fit_js[primitive_key] if primitive_length == 'movej_fit' else self.step_curve_fit[primitive_key]

        # Calculate the chosen primitive
        new_primitive_length = np.ceil(len(old_primitive_fit) * primitive_length).astype(int)
        new_primitive_length = max(new_primitive_length, 2)

        # Re-fit
        cur_idx = self.breakpoints[-1]
        next_idx = cur_idx + new_primitive_length
        new_curve_fit, new_curve_fit_R, new_curve_fit_js, max_error, max_ori_error = self.primitives[primitive_key](self.curve[cur_idx:next_idx], self.curve_js[cur_idx:next_idx], self.curve_R[cur_idx:next_idx])
        if primitive_key == 'movec_fit' and np.linalg.norm(new_curve_fit[-1] - new_curve_fit[0]) < 50:
            primitive_key = 'movel_fit'
        # if primitive_key == 'movec_fit' and len(self.step_curve_fit[primitive_key]) < 5:
        #     primitive_key = 'movel_fit'

        # Add the new primitive to curve fit
        self.primitive_choices.append(primitive_key)
        self.max_error = max(self.max_error, max_error)
        self.max_ori_error = max(self.max_ori_error, max_ori_error)

        self.primitive_fits.append((primitive_key, new_curve_fit))
        self.breakpoints.append(min(self.breakpoints[-1] + len(new_curve_fit), len(self.curve)))
        self.curve_fit.extend(new_curve_fit)
        self.curve_fit_R.extend(new_curve_fit_R)
        if primitive_key == 'movej_fit':
            self.curve_fit_js.extend(new_curve_fit_js)
        else:
            new_curve_fit_js = self.car2js(new_curve_fit, new_curve_fit_R)
            self.curve_fit_js.extend(new_curve_fit_js)

        if primitive_key == 'movec_fit':
            if len(new_curve_fit) == 2:
                self.points.append([new_curve_fit[0], new_curve_fit[1]])
                self.q_bp.append([new_curve_fit_js[0], new_curve_fit_js[1]])
            else:
                self.points.append([new_curve_fit[int(len(new_curve_fit)/2)], new_curve_fit[-1]])
                self.q_bp.append([new_curve_fit_js[int(len(new_curve_fit_js)/2)], new_curve_fit_js[-1]])
                assert np.linalg.norm(self.points[-1][1] - self.points[-1][0]) > 0
        elif primitive_key == 'movel_fit':
            self.points.append([new_curve_fit[-1]])
            self.q_bp.append([new_curve_fit_js[-1]])
        else:
            self.points.append([new_curve_fit[-1]])
            self.q_bp.append([new_curve_fit_js[-1]])

        # ==========
        # RL Part
        done = self.breakpoints[-1] >= len(self.curve) - 1
        reward = self.reward_function(done)
        state = None

        if not done:
            remaining_curve = self.curve[self.breakpoints[-1]:, :]
            normalized_curve = PCA_normalization(remaining_curve)
            self.greedy_primitives_fit()
            primitive_features = np.array([self.step_max_error_place['movel_fit'], self.step_max_error_place['movej_fit'], self.step_max_error_place['movec_fit']])
            primitive_features = np.concatenate([primitive_features, self.step_slope_diff_js['movel_fit'], self.step_slope_diff_js['movej_fit'], self.step_slope_diff_js['movec_fit']])
            state = (normalized_curve, primitive_features)
        else:
            self.curve_fit = np.array(self.curve_fit)
            self.curve_fit_R = np.array(self.curve_fit_R)
            self.curve_fit_js = np.array(self.curve_fit_js)
        return state, reward, done

    def reward_function(self, done):
        if not done:
            return 0
        else:
            self.exec_time, self.exec_max_error, self.exec_max_normal_error = self.robot_studio_execute()
            reward = 10 * (3 - self.exec_time)
            return reward

    def robot_studio_execute(self, verbose=False):
        ms = MotionSend()
        breakpoints, primitives_choices, points, q_bp = self.breakpoints, self.primitive_choices, self.points, self.q_bp

        ############insert initial configuration#################
        primitives_choices.insert(0, 'movej_fit')
        points.insert(0, [self.curve_fit[0]])
        q_bp.insert(0, [self.curve_fit_js[0]])

        str_points = []
        str_q_points = []
        for p in points:
            str_points.append(str(p))
        for q in q_bp:
            str_q_points.append(str(q))
        df = pd.DataFrame({'breakpoints': breakpoints, 'primitives': primitives_choices, 'points': str_points, 'q_points': str_q_points})
        df.to_csv('circle_uncertain_commands.csv', index=False)


        ###extension
        points, q_bp = ms.extend(self.robot, q_bp, primitives_choices, breakpoints, points)

        logged_data = ms.exec_motions(self.robot, primitives_choices, breakpoints, points, q_bp, v500, z10)
        StringData = StringIO(logged_data)
        df = read_csv(StringData, sep=",")

        ##############################train_data analysis#####################################
        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = ms.logged_data_analysis(self.robot, df, realrobot=False)

        #############################chop extension off##################################
        lam, curve_exe, curve_exe_R, curve_exe_js, speed, timestamp = ms.chop_extension(curve_exe, curve_exe_R,
                                                                                        curve_exe_js, speed, timestamp,
                                                                                        self.curve[0, :3],
                                                                                        self.curve[-1, :3])
        error, angle_error = calc_all_error_w_normal(curve_exe, self.curve, curve_exe_R[:, :, -1],
                                                     self.curve_R[:, :, -1])
        execution_time, max_error, normal_error = timestamp[-1] - timestamp[0], np.max(error), np.max(angle_error)
        if verbose:
            print('time: ', timestamp[-1] - timestamp[0], 'error: ', np.max(error), 'normal error: ', np.max(angle_error))

        return execution_time, max_error, normal_error
