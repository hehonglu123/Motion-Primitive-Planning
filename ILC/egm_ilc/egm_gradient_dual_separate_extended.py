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
	alpha_default=1.

	dataset='curve_2/'
	data_dir="../../data/"+dataset
	solution_dir=data_dir+'dual_arm/'+'diffevo_pose6/'
	
	robot1=robot_obj('ABB_6640_180_255','../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='')
	robot2=robot_obj('ABB_1200_5_90','../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',base_transformation_file=solution_dir+'base.csv',acc_dict_path='')

	relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,data_dir,solution_dir,robot1,robot2)


	egm1 = rpi_abb_irc5.EGM(port=6510)
	egm2 = rpi_abb_irc5.EGM(port=6511)

	et1=EGM_toolbox(egm1,robot1)
	et2=EGM_toolbox(egm2,robot2)

	vd=2222
	idx=et1.downsample24ms(relative_path,vd)
	extension_start=100
	extension_end=100

	curve_cmd_js1=curve_js1[idx]
	curve_cmd_js2=curve_js2[idx]
	
	##add extension
	curve_cmd_js1_ext=et1.add_extension_egm_js(curve_cmd_js1,extension_start=extension_start,extension_end=extension_end)
	curve_cmd_js2_ext=et1.add_extension_egm_js(curve_cmd_js2,extension_start=extension_start,extension_end=extension_end)

	curve_js1_d=copy.deepcopy(curve_cmd_js1_ext)
	curve_js2_d=copy.deepcopy(curve_cmd_js2_ext)

	
	iteration=100
	skip=False

	for i in range(iteration):
		if skip:
			timestamp1=timestamp1_temp
			timestamp2=timestamp2_temp
			curve_exe_js1=curve_exe_js1_temp
			curve_exe_js2=curve_exe_js2_temp

			error=error_temp
			angle_error=angle_error_temp
			speed=speed_temp
		else:
			################################traverse curve for both arms#####################################

			###jog both arm to start pose
			et1.jog_joint(curve_cmd_js1_ext[0])
			timestamp1,curve_exe_js1=et1.traverse_curve_js(curve_cmd_js1_ext)

			et2.jog_joint(curve_cmd_js2_ext[0])
			timestamp2,curve_exe_js2=et2.traverse_curve_js(curve_cmd_js2_ext)

			##############################calcualte error########################################
			_,_,_,_,relative_path_exe,relative_path_exe_R=form_relative_path(curve_exe_js1[extension_start:-extension_end],curve_exe_js2[extension_start:-extension_end],robot1,robot2)

			lam=calc_lam_cs(relative_path_exe)
			speed=np.gradient(lam)/np.gradient(timestamp1[extension_start:-extension_end])
			error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
		
		print(max(error))


		##############################PUSH in ERROR DIRECTION########################################
		alpha=0.5
		error1=curve_js1_d-curve_exe_js1
		error2=curve_js2_d-curve_exe_js2

		curve_cmd_js1_ext=curve_cmd_js1_ext+alpha*error1
		curve_cmd_js2_ext=curve_cmd_js2_ext+alpha*error2
		# print('step size: ',alpha)
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
		h1, l1 = ax1.get_legend_handles_labels()
		h2, l2 = ax2.get_legend_handles_labels()
		ax1.legend(h1+h2, l1+l2, loc=1)
		# plt.show()
		plt.savefig('recorded_data/iteration '+str(i))
		plt.clf()

		plt.plot(error1,label=['joint1','joint2','joint3','joint4','joint5','joint6'])
		plt.legend()
		plt.savefig('recorded_data/iteration '+str(i)+'_r1')
		plt.clf()

		plt.plot(error2,label=['joint1','joint2','joint3','joint4','joint5','joint6'])
		plt.legend()
		plt.savefig('recorded_data/iteration '+str(i)+'_r2')
		plt.clf()

		###save EGM commands
		df=DataFrame({'q0':curve_cmd_js1_ext[:,0],'q1':curve_cmd_js1_ext[:,1],'q2':curve_cmd_js1_ext[:,2],'q3':curve_cmd_js1_ext[:,3],'q4':curve_cmd_js1_ext[:,4],'q5':curve_cmd_js1_ext[:,5]})
		df.to_csv('recorded_data/EGM_arm1.csv',header=False,index=False)
		df=DataFrame({'q0':curve_js1_d[:,0],'q1':curve_js1_d[:,1],'q2':curve_js1_d[:,2],'q3':curve_js1_d[:,3],'q4':curve_js1_d[:,4],'q5':curve_js1_d[:,5]})
		df.to_csv('recorded_data/EGM_arm1_d.csv',header=False,index=False)
		df=DataFrame({'q0':curve_cmd_js2_ext[:,0],'q1':curve_cmd_js2_ext[:,1],'q2':curve_cmd_js2_ext[:,2],'q3':curve_cmd_js2_ext[:,3],'q4':curve_cmd_js2_ext[:,4],'q5':curve_cmd_js2_ext[:,5]})
		df.to_csv('recorded_data/EGM_arm2.csv',header=False,index=False)
		df=DataFrame({'q0':curve_js2_d[:,0],'q1':curve_js2_d[:,1],'q2':curve_js2_d[:,2],'q3':curve_js2_d[:,3],'q4':curve_js2_d[:,4],'q5':curve_js2_d[:,5]})
		df.to_csv('recorded_data/EGM_arm2_d.csv',header=False,index=False)

if __name__ == '__main__':
	main()