import os
import numpy as np

from toolbox_circular_fit import *
import pandas as pd

# sys.path.append('../toolbox')
# from robot_def import *
from collections import namedtuple

# Point3D = namedtuple("Point3D", ('x', 'y', 'z'))
BP_Feature = namedtuple("BP_Feature", ('longest_moveL', 'longest_moveC', 'last_bp'))
ERROR_THRESHOLD = 1.0


def read_data(dir_path, max_curves=float('inf')):
    all_data = []
    col_names = ['X', 'Y', 'Z', 'direction_x', 'direction_y', 'direction_z']
    count = 0

    for file in os.listdir(dir_path):
        count += 1
        if count > max_curves:
            break
        file_path = dir_path + os.sep + file
        data = pd.read_csv(file_path, names=col_names)
        curve_x = data['X'].tolist()
        curve_y = data['Y'].tolist()
        curve_z = data['Z'].tolist()
        curve = np.vstack((curve_x, curve_y, curve_z)).T
        all_data.append(curve)

    return all_data


class Primitive(object):
    id_counter = 0

    def __init__(self, start, end, move_type='L'):
        self.move_type = move_type
        self.start = start
        self.end = end
        Primitive.id_counter += 1
        self.id = Primitive.id_counter
        self.curve = []
        self.ref_curve = []

    def __lt__(self, other):
        return len(self.curve) < len(other.curve)

    def __le__(self, other):
        return len(self.curve) <= len(other.curve)

    def __gt__(self, other):
        return len(self.curve) > len(other.curve)

    def __ge__(self, other):
        return len(self.curve) >= len(other.curve)

    def __eq__(self, other):
        return len(self.curve) == len(other.curve)

    def curve_fit(self, curve, p=[]):
        if self.move_type == 'L':
            return self.move_l_fit(curve)
        elif self.move_type == 'C':
            return self.move_c_fit(curve, p)
        elif self.move_type == 'J':
            return self.move_j_fit(curve)

    def move_l_fit(self, curve, p=[]):
        if len(p) == 0:
            A = np.vstack((np.ones(len(curve)), np.arange(0, len(curve)))).T
            b = curve
            res = np.linalg.lstsq(A, b, rcond=None)[0]
            start_point = res[0]
            slope = res[1].reshape(1, -1)
            ###with constraint point
        else:
            A = np.arange(0, len(curve)).reshape(-1, 1)
            b = curve - curve[0]
            res = np.linalg.lstsq(A, b, rcond=None)[0]
            slope = res.reshape(1, -1)
            start_point = p

        curve_fit = np.dot(np.arange(0, len(curve)).reshape(-1, 1), slope) + start_point

        max_error = np.max(np.linalg.norm(curve - curve_fit, axis=1))

        self.curve = curve_fit

        return curve_fit, max_error

    def move_c_fit(self, curve, p=[]):
        curve_fit, curve_fit_circle = circle_fit(curve, p)
        max_error = np.max(np.linalg.norm(curve - curve_fit, axis=1))
        self.curve = curve_fit

        return curve_fit, max_error

    def move_j_fit(self, curve):
        pass


class BreakPoint(object):
    def __init__(self, idx=0, x=0, y=0, z=0, position=None):
        self.left = None
        self.right = None
        self._position = np.array([x, y, z]) if position is None else position
        self._features = BP_Feature(None, None, None)
        self.idx = idx

    def set_features(self, features: BP_Feature):
        # self._features.longest_moveC = features.longest_moveC
        # self._features.longest_moveL = features.longest_moveL
        # self._features.last_bp = features.last_bp
        self._features = features

    def get_features(self):
        return self._features

    def is_end(self):
        return self.left is None or self.right is None

    def __eq__(self, other):
        return self.idx == other.idx
