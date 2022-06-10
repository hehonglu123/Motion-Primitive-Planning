import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv, read_excel
import sys
sys.path.append('../../../toolbox')
from abb_motion_program_exec_client import *
from robots_def import *
from lambda_calc import *
from error_check import *

def norm_vec(v):
    return v/np.linalg.norm(v)

robot = m900ia(d=50)

save_folder='../data/data_collect/'
data_folder=save_folder+'scene_1/'

