import numpy as np

from toolbox.MotionSend import MotionSend
from toolbox.toolbox_circular_fit import arc_from_3point
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
        ms = MotionSend()
        p_bp, q_bp = ms.extend(self.robot, self.q_bp, self.primitives, self.breakpoints, self.p_bp)
        self.p_bp = p_bp
        self.q_bp = q_bp
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

    def extend(self):
        # initial point extension
        pose_start = self.robot.fwd(self.q_bp[0][-1])
        p_start = pose_start.p
        R_start = pose_start.R
        pose_end = self.robot.fwd(self.q_bp[1][-1])
        p_end = pose_end.p
        R_end = pose_end.R
