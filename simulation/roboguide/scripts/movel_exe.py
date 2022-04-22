from math import radians
import numpy as np
from pandas import read_csv

from general_robotics_toolbox import *
import sys

from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../toolbox')
from robots_def import *
sys.path.append('fanuc_toolbox')
from fanuc_client import *

def main():
    
    # define m900ia
    robot = m900ia(d=50)

if __name__=='__main__':
    main()

