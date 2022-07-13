import os
import pandas as pd
import numpy as np

from rl_ilc_env import ILCEnv
from ilc_replayer import Replayer
from td3_agent import TD3Agent


def read_data(curve_idx, data_dir):
    base_path = data_dir + "base/curve_{}.csv".format(curve_idx)
    js_path = data_dir + "js/curve_{}.csv".format(curve_idx)

    base_data = pd.read_csv(base_path, header=None).values
    js_data = pd.read_csv(js_path, header=None).values

    curve = base_data[:, :3]
    curve_normal = base_data[:, 3:]
    curve_js = js_data

    return curve, curve_normal, curve_js


def train(agent: TD3Agent, data_dir, max_episode=10000):
    forward_data_dir = data_dir + os.sep + 'forward' + os.sep
    reverse_data_dir = data_dir + os.sep + 'reverse' + os.sep

    for episode in range(max_episode):
        curve_idx = np.random.randint(100)
        data_dir = forward_data_dir if episode % 2 == 0 else reverse_data_dir
        curve, curve_normal, curve_js = read_data(curve_idx, data_dir)


