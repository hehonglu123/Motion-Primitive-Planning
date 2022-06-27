import numpy as np
import pandas as pd
import random
import time

from collections import namedtuple
import torch

from rl_fit.archive.trajectory_utils import Primitive, BreakPoint, read_base_data

from sklearn.decomposition import PCA

# sys.path.append('../greedy_fitting/')
# from greedy_simplified import greedy_fit

# sys.path.append('../toolbox')
# from robots_def import *

Curve_Error = namedtuple("Curve_Error", ('primitive', 'max_error'))
State = namedtuple("State", ("longest_type", "curve_features"))
Action = namedtuple("Action", ("Type", "Length", "Code"))
Memory = namedtuple('Memory', ('state', 'action', 'reward', 'next_state', 'done'))
primitive_type_code = {"L": 0, "C": 1, "J": 2}

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

ERROR_THRESHOLD = 1.0
MAX_GREEDY_STEP = 10

EPS_START = 0.5
EPS_END = 0.01
EPS_DECAY = 10

TAU_START = 10
TAU_END = 0.01
MAX_TAU_EPISODE = 0.6
LEARNING_RATE = 0.001
LEARNING_FREQ = 4
GAMMA = 0.99
BATCH_SIZE = 16

REWARD_FINISH = 2000  # -------------------------------------- HERE
REWARD_DECAY_FACTOR = 0.9
REWARD_STEP = -10

SAVE_MODEL = 10

# axis1_range = []
# axis2_range = []
# axis3_range = []
# length = []
# pca_curve_count = 0


def episode_tau(i_episode, n_episode):
    # factor = min(1, i_episode/(n_episode * MAX_TAU_EPISODE))
    factor = min(1, i_episode / 200)
    tau = TAU_START - factor * (TAU_START - TAU_END)
    return tau


def greedy_data_to_dict(greedy_data: pd.DataFrame):
    ret_dict = dict()
    for index, row in greedy_data.iterrows():
        ret_dict[row['id']] = row['n_primitives']
    return ret_dict


def pca_circular_check(curve):
    pca = PCA(n_components=3)
    curve_pca = pca.fit_transform(curve)

    if np.max(curve_pca[:,2]) - np.min(curve_pca[:,2]) <= ERROR_THRESHOLD * 2:
        return True
    return False


def greedy_fit_primitive(last_bp, curve, p):
    longest_fits = {'C': None, 'L': None}

    search_left_c = search_left_l = last_bp
    search_right_c = search_right_l = len(curve)
    search_point_c = min(search_left_c+2000, search_right_c)
    search_point_l = min(search_left_l+2000, search_right_l)

    stationary_c = False
    stationary_l = False

    step_count = 0
    while step_count < MAX_GREEDY_STEP:
        step_count += 1

        # moveC
        left_bp = BreakPoint(idx=last_bp)
        right_bp = BreakPoint(idx=search_point_c)
        target_curve = curve[last_bp:max(last_bp+1, int(search_point_c) + 1)]
        # print(len(target_curve), last_bp, search_point_c)

        if len(target_curve) >= 3 and not stationary_c:
            if not pca_circular_check(target_curve):
                search_right_c = search_point_c
                new_search_point_c = np.floor(np.mean([search_left_c, search_point_c]))
            else:
                primitive_c = Primitive(start=left_bp, end=right_bp, move_type='C')
                # print("---", len(target_curve), last_bp, search_point_c, step_count)
                curve_fit, max_error = primitive_c.curve_fit(target_curve, p=p)
                if max_error <= ERROR_THRESHOLD:
                    if longest_fits['C'] is None or primitive_c > longest_fits['C'].primitive:
                        longest_fits['C'] = Curve_Error(primitive_c, max_error)

                    search_left_c = search_point_c
                    new_search_point_c = np.floor(np.mean([search_point_c, search_right_c]))
                else:
                    search_right_c = search_point_c
                    new_search_point_c = np.floor(np.mean([search_left_c, search_point_c]))
            stationary_c = new_search_point_c == search_point_c
            search_point_c = new_search_point_c

        # moveL
        if not stationary_l:
            left_bp = BreakPoint(idx=last_bp)
            right_bp = BreakPoint(idx=search_point_l)
            target_curve = curve[last_bp:max(last_bp+1, int(search_point_l) + 1)]

            primitive_l = Primitive(start=left_bp, end=right_bp, move_type='L')
            curve_fit, max_error = primitive_l.curve_fit(target_curve)
            if max_error <= ERROR_THRESHOLD:
                if longest_fits['L'] is None or primitive_l > longest_fits['L'].primitive:
                    longest_fits['L'] = Curve_Error(primitive_l, max_error)

                search_left_l = search_point_l
                new_search_point_l = np.floor(np.mean([search_point_l, search_right_l]))
            else:
                search_right_l = search_point_l
                new_search_point_l = np.floor(np.mean([search_left_l, search_point_l]))
            stationary_l = new_search_point_l == search_point_l
            search_point_l = new_search_point_l

    longest_type = 'L'
    if longest_fits['L'] is None or (longest_fits['C'] is not None and
                                     longest_fits['C'].primitive > longest_fits['L'].primitive):
        longest_type = 'C'
    elif longest_fits['C'] is None or (longest_fits['L'] is not None and
                                       longest_fits['L'].primitive >= longest_fits['C'].primitive):
        longest_type = 'L'

    return longest_fits, longest_type


def greedy_fit(curve_base_data):
    print("Greedy Fitting Start")
    greedy_curve_idx = []
    greedy_steps = []
    for i, (curve, _) in enumerate(curve_base_data):
        timer = time.time_ns()
        step_count = 0
        last_point = 0
        fitted_curve = [curve[0]]
        while len(fitted_curve) < len(curve):
            # print(step_count)
            longest_fits, longest_type = greedy_fit_primitive(last_point, curve, fitted_curve[-1])

            # if longest_type == 'C':
            #     move_c_primitive = longest_fits['C'].primitive.curve
            #     print("MoveC", last_point)
            #     if not pca_circular_check(move_c_primitive):
            #         print("PCA check fail")
            #         # pca_curve_count += 1
            #         pca = PCA(n_components=3)
            #         curve_pca = pca.fit_transform(curve)
            #         x1 = np.max(curve_pca[:,0]) - np.min(curve_pca[:,0])
            #         x2 = np.max(curve_pca[:,1]) - np.min(curve_pca[:,1])
            #         x3 = np.max(curve_pca[:,2]) - np.min(curve_pca[:,2])
            #         axis1_range.append(x1)
            #         axis2_range.append(x2)
            #         axis3_range.append(x3)

            fit_primitive = longest_fits[longest_type].primitive.curve
            fitted_curve = np.concatenate([fitted_curve, fit_primitive])
            last_point = len(fitted_curve) - 1
            step_count += 1
        greedy_curve_idx.append(i)
        greedy_steps.append(step_count)
        print("{} / {} --- {:.2f}s --- {} steps".format(i, len(curve_base_data), (time.time_ns() - timer)*1e-9, step_count))
    greedy_df = pd.DataFrame({"id": greedy_curve_idx, "n_primitives": greedy_steps})
    greedy_df.to_csv("data/new_greedy_data.csv", index=False)


def save_data(episode_rewards, episode_steps, episode_target_curve):
    df = pd.DataFrame({"episode_rewards": episode_rewards,
                       "episode_steps": episode_steps,
                       "curve": episode_target_curve})
    df.to_csv("Training Curve Data.csv")


def save_evaluation_data(episode_steps, episode_target_curve):
    df = pd.DataFrame({"num_primitives": episode_steps,
                       "curve": episode_target_curve})
    df.to_csv("Evaluate Curve Data.csv")


def main():
    curve_base_data = read_base_data("data/base")
    # curve_js_data = read_js_data("data/js")
    greedy_fit(curve_base_data)
    # pca_df = pd.DataFrame({"axis1": axis1_range, "axis2": axis2_range, "axis3": axis3_range})
    # pca_df.to_csv("pca_movec.csv", index=False)


if __name__ == '__main__':
    main()

