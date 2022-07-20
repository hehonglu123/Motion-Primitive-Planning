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

	robot1=abb6640(d=50)
	with open(data_dir+'dual_arm/abb1200.yaml') as file:
		H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	with open(data_dir+'dual_arm/tcp.yaml') as file:
		H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

	egm1 = rpi_abb_irc5.EGM(port=6510)
	egm2 = rpi_abb_irc5.EGM(port=6511)

	et1=EGM_toolbox(egm1,robot1)
	et2=EGM_toolbox(egm2,robot2)

	curve_js1=read_csv(data_dir+"dual_arm/arm1.csv",header=None).values
	curve_js2=read_csv(data_dir+"dual_arm/arm2.csv",header=None).values
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	vd=500
	idx=et1.downsample24ms(relative_path,vd)

	curve_cmd_js1=curve_js1[idx]
	curve_cmd_js2=curve_js2[idx]
	curve_js1_d=curve_js1[idx]
	curve_js2_d=curve_js2[idx]

	extension_num=66
	iteration=100
	adjust_weigt_it=20

	weights=np.ones(len(curve_cmd_js1))
	for i in range(iteration):

		##add extension
		curve_cmd_js1_ext=et1.add_extension_egm_js(curve_cmd_js1,extension_num=extension_num)
		curve_cmd_js2_ext=et1.add_extension_egm_js(curve_cmd_js2,extension_num=extension_num)


		###jog both arm to start pose
		
		
		################################traverse curve for both arms#####################################

		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_ext[0])
		timestamp1,curve_exe_js1=et1.traverse_curve_js(curve_cmd_js1_ext)

		et2.jog_joint(curve_cmd_js2_ext[0])
		timestamp2,curve_exe_js2=et2.traverse_curve_js(curve_cmd_js2_ext)

		##############################calcualte error########################################
		relative_path_exe,relative_path_exe_R=form_relative_path(robot1,robot2,curve_exe_js1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay],curve_exe_js2[extension_num+et1.idx_delay:-extension_num+et1.idx_delay],base2_R,base2_p)
		lam=calc_lam_cs(relative_path_exe)
		speed=np.gradient(lam)/np.gradient(timestamp1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay])
		error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
		print(max(error))

		if i>adjust_weigt_it:
			print('weights adjusted')
			weights[np.where(error>0.5)]=5

		##############################ILC########################################
		error1=curve_exe_js1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_js1_d
		error1=error1*weights[:, np.newaxis]
		error1_flip=np.flipud(error1)
		# print(error1)
		###calcualte agumented input
		curve_cmd_js1_aug=curve_cmd_js1+error1_flip

		error2=curve_exe_js2[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_js2_d
		error2=error2*weights[:, np.newaxis]
		error2_flip=np.flipud(error2)
		###calcualte agumented input
		curve_cmd_js2_aug=curve_cmd_js2+error2_flip

		##add extension
		curve_cmd_js1_ext_aug=et1.add_extension_egm_js(curve_cmd_js1_aug,extension_num=extension_num)
		curve_cmd_js2_ext_aug=et1.add_extension_egm_js(curve_cmd_js2_aug,extension_num=extension_num)


		################################traverse curve for both arms#####################################

		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_ext_aug[0])
		_,curve_exe_js1_aug=et1.traverse_curve_js(curve_cmd_js1_ext_aug)

		et2.jog_joint(curve_cmd_js2_ext_aug[0])
		_,curve_exe_js2_aug=et2.traverse_curve_js(curve_cmd_js2_ext_aug)




		###get new error
		delta1_new=curve_exe_js1_aug[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_exe_js1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]
		delta1_new=delta1_new
		grad1=np.flipud(delta1_new)

		alpha=0.5
		curve_cmd_js1=curve_cmd_js1-alpha*grad1

		delta2_new=curve_exe_js2_aug[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_exe_js2[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]
		delta2_new=delta2_new
		grad2=np.flipud(delta2_new)

		alpha=0.5
		curve_cmd_js2=curve_cmd_js2-alpha*grad2

		##############################plot error#####################################

		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam, speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		ax2.axis(ymin=0,ymax=30)
		ax1.axis(ymin=0,ymax=1.2*vd)

		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
		plt.title("Speed and Error Plot")
		ax1.legend(loc=0)

		ax2.legend(loc=0)

		plt.legend()
		# plt.show()
		plt.savefig('iteration '+str(i))
		plt.clf()


		plt.plot(error1)
		plt.savefig('iteration '+str(i)+'_r1')
		plt.clf()


		plt.plot(error2)
		plt.savefig('iteration '+str(i)+'_r2')
		plt.clf()


if __name__ == '__main__':
	main()