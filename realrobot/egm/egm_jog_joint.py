import numpy as np
import time, sys
from pandas import *

sys.path.append('../../toolbox')
sys.path.append('../../toolbox/egm_toolbox')

from robots_def import *
from error_check import *
from lambda_calc import *
from EGM_toolbox import *


def main():
	##########6640 cartesian jog test##############################
	robot=abb6640(d=50)
	egm = rpi_abb_irc5.EGM(port=6510)
	et=EGM_toolbox(egm,robot)
	et.jog_joint(0.*np.ones(6))
	##########1200 joint jog test##############################
	# robot=abb1200()
	# egm = rpi_abb_irc5.EGM(port=6511)
	# et=EGM_toolbox(egm,robot)
	# et.jog_joint(0.3*np.ones(6))

if __name__ == '__main__':
	main()