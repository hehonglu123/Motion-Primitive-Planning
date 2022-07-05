import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import RobotRaconteurCompanion as RRC

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys, os
from io import StringIO
from scipy.signal import find_peaks
import matplotlib.animation as animation

# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
from ilc_toolbox import *
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
from blending import *


class max_grad(object):
	def __init__(self):

def main():
	with RR.ServerNodeSetup("plt_service",12180) as node_setup:
		RRC.RegisterStdRobDefServiceTypes(RRN)
		RRN.RegisterServiceTypeFromFile("edu.rpi.robotics.plt_live")

if __name__ == '__main__':
	main()