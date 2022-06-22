import numpy as np
import torch


from greedy_fitting.greedy import greedy_fit
from toolbox.robots_def import *
from general_robotics_toolbox import *
from toolbox.MotionSend import *
from rl_fit.utils.curve_normalization import PCA_normalization

class MotionPrimitiveEnv(greedy_fit):
    def __init__(self, args):
        super(MotionPrimitiveEnv, self).__init__(args.robot, args.curve_js, args.max_error_threshold, args.max_ori_threshold)

        self.action_dim = args.discrete_actions * 3
        self.robot = args.robot
        self.curve_js = args.curve_js

        self.step_greedy_primitives = {'movel_fit': None, 'movej_fit': None, 'movec_fit': None}
        self.step_curve_fit = {}
        self.step_curve_fit_R = {}
        self.step_curve_fit_js = {}
        self.step_max_error = {}
        self.step_max_ori_error = {}
        self.step_max_error_place = {}
        self.step_slope_diff = {}

        self.max_error = 999
        self.max_ori_error = 999
        self.primitive_choices = []
        self.points = []

        self.exec_time = -1
        self.exec_max_error = -1
        self.exec_max_error_angle = -1

        self.error_curve = None
        self.normal_error_curve = None

    def greedy_primitives_fit(self, cur_idx):
        for key in self.primitives:
            self.step_curve_fit[key], self.step_curve_fit_R[key], self.step_curve_fit_js[key], self.step_max_error[key], self.step_max_ori_error[key] =


    