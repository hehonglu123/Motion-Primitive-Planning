import numpy as np
import time, sys
from pandas import *

sys.path.append('../../../toolbox')
sys.path.append('../../../toolbox/egm_toolbox')

from robots_def import *
from error_check import *
from lambda_calc import *
from EGM_toolbox import *


def main():
	##########6640 cartesian jog test##############################
	robot=abb6640(d=50)
	egm = rpi_abb_irc5.EGM(port=6510)
	et=EGM_toolbox(egm,robot)

	p=[1500,0,1000]
	R=np.array([[-1,0,0],
					[0,1,0],
					[0,0,-1]])
	et.jog_joint_cartesian(p,R)
	##########1200 cartesian jog test##############################
	# robot=abb1200()
	# egm = rpi_abb_irc5.EGM(port=6511)
	# et=EGM_toolbox(egm,robot)

	# p=[1000,0, 536.5520629882812]
	# R=np.array([[0,0,1],
	# 			[0,1,0],
	# 			[-1,0,0]])
	# et.jog_joint_cartesian(p,R=q2R([0.25881898403167725,0.0,0.9659258127212524,0.0]))
	##########1200 joint jog test##############################
	robot=abb1200()
	egm = rpi_abb_irc5.EGM(port=6511)
	et=EGM_toolbox(egm,robot)
	et.jog_joint(np.ones(6))

if __name__ == '__main__':
	main()