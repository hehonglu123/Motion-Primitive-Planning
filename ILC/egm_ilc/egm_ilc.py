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
	robot=abb6640(d=50)
	egm = rpi_abb_irc5.EGM()

	# robot=abb1200()
	# egm = rpi_abb_irc5.EGM(port=6511)
	et=EGM_toolbox(egm,robot)

	dataset='wood/'
	data_dir='../../data/'
	curve_js = read_csv(data_dir+dataset+'dual_arm/arm1.csv',header=None).values
	curve = read_csv(data_dir+dataset+'Curve_in_base_frame.csv',header=None).values

	vd=250
	idx=et.downsample24ms(curve,vd)

	curve_cmd_js=curve_js[idx]
	curve_js_d=curve_js[idx]

	extension_num=66
	iteration=50
	for i in range(iteration):

		##add extension
		curve_cmd_js_ext=et.add_extension_egm_js(curve_cmd_js,extension_num=extension_num)


		###jog both arm to start pose
		et.jog_joint(curve_cmd_js_ext[0])
		
		###traverse the curve
		timestamp,curve_exe_js=et.traverse_curve_js(curve_cmd_js_ext)
		###############################ILC########################################
		error=curve_exe_js[extension_num+et.idx_delay:-extension_num+et.idx_delay]-curve_js_d
		error_flip=np.flipud(error)
		###calcualte agumented input
		curve_cmd_js_aug=curve_cmd_js+error_flip

		time.sleep(1)

		##add extension
		curve_cmd_js_ext_aug=et.add_extension_egm_js(curve_cmd_js_aug,extension_num=extension_num)
		###move to start first
		print('moving to start point')
		et.jog_joint(curve_cmd_js_ext_aug[0])
		###traverse the curve
		timestamp_aug,curve_exe_js_aug=et.traverse_curve_js(curve_cmd_js_ext_aug)

		###get new error
		delta_new=curve_exe_js_aug[extension_num+et.idx_delay:-extension_num+et.idx_delay]-curve_exe_js[extension_num+et.idx_delay:-extension_num+et.idx_delay]

		grad=np.flipud(delta_new)

		alpha=0.5
		curve_cmd_js=curve_cmd_js-alpha*grad

		##############################plot error#####################################
		# plt.plot(error)
		# plt.show()

	DataFrame(curve_cmd_js).to_csv('dual_arm/egm_arm2.csv',header=False,index=False)

if __name__ == "__main__":
	main()