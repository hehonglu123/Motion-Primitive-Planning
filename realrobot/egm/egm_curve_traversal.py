import numpy as np
from general_robotics_toolbox import *
import sys, threading
sys.path.append('../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
sys.path.append('../../toolbox/egm_toolbox')
from EGM_toolbox import *
import rpi_abb_irc5


def main():
	data_dir='../../data/wood/'

	robot=abb6640(d=50)
	egm = rpi_abb_irc5.EGM(port=6510)
	curve_js=read_csv(data_dir+"dual_arm/arm1.csv",header=None).values

	# robot=abb1200()
	# egm = rpi_abb_irc5.EGM(port=6511)
	# curve_js=read_csv(data_dir+"dual_arm/arm2.csv",header=None).values

	et=EGM_toolbox(egm,robot)
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	vd=50
	idx=et.downsample24ms(relative_path,vd)
	extension_start=40
	extension_end=10

	curve_cmd_js=curve_js[idx]
	
	##add extension
	curve_cmd_js_ext=et.add_extension_egm_js(curve_cmd_js,extension_start=extension_start,extension_end=extension_end)
	################################traverse curve for both arms#####################################

	###jog both arm to start pose
	et.jog_joint(curve_cmd_js_ext[0])
	timestamp,curve_exe_js=et.traverse_curve_js(curve_cmd_js_ext)

if __name__ == '__main__':
	main()