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
	idx_delay=int(et1.delay/et1.ts)


	vd=2000
	idx=et1.downsample24ms(relative_path,vd)
	extension_start=100
	extension_end=idx_delay+1

	curve_cmd_js1=curve_js1[idx]
	curve_cmd_js2=curve_js2[idx]

	curve_js1_d=curve_js1[idx]
	curve_js2_d=curve_js2[idx]
	
	

	#################FIRST TWO ITERATION, PUSH IN ERROR DIRECTION#########################################################
	for i in range(2):
		##add extension
		curve_cmd_js1_ext=et1.add_extension_egm_js(curve_cmd_js1,extension_start=extension_start,extension_end=extension_end)
		curve_cmd_js2_ext=et1.add_extension_egm_js(curve_cmd_js2,extension_start=extension_start,extension_end=extension_end)

		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_ext[0])
		timestamp1,curve_exe_js1=et1.traverse_curve_js(curve_cmd_js1_ext)

		et2.jog_joint(curve_cmd_js2_ext[0])
		timestamp2,curve_exe_js2=et2.traverse_curve_js(curve_cmd_js2_ext)

		error1=curve_js1_d-curve_exe_js1[extension_start+idx_delay:-extension_end+idx_delay]
		curve_cmd_js1=curve_cmd_js1+error1
		error2=curve_js2_d-curve_exe_js2[extension_start+idx_delay:-extension_end+idx_delay]
		curve_cmd_js2=curve_cmd_js2+error2


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
			curve_cmd_js1_ext=et1.add_extension_egm_js(curve_cmd_js1,extension_start=extension_start,extension_end=extension_end)
			curve_cmd_js2_ext=et1.add_extension_egm_js(curve_cmd_js2,extension_start=extension_start,extension_end=extension_end)
			
			###jog both arm to start pose
			et1.jog_joint(curve_cmd_js1_ext[0])
			timestamp1,curve_exe_js1=et1.traverse_curve_js(curve_cmd_js1_ext)

			et2.jog_joint(curve_cmd_js2_ext[0])
			timestamp2,curve_exe_js2=et2.traverse_curve_js(curve_cmd_js2_ext)

			##############################calcualte error########################################
			_,_,_,_,relative_path_exe,relative_path_exe_R=form_relative_path(curve_exe_js1[extension_start+idx_delay:-extension_end+idx_delay],curve_exe_js2[extension_start+idx_delay:-extension_end+idx_delay],robot1,robot2)

			lam=calc_lam_cs(relative_path_exe)
			speed=np.gradient(lam)/np.gradient(timestamp1[extension_start+idx_delay:-extension_end+idx_delay])
			error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
		
		print(max(error))


		##############################ILC########################################
		gradient_step=1
		error1=curve_exe_js1[extension_start+idx_delay:-extension_end+idx_delay]-curve_js1_d
		error1=error1
		error1_flip=np.flipud(error1)
		# print(error1)
		###calcualte agumented input
		curve_cmd_js1_aug=curve_cmd_js1+error1_flip*gradient_step

		error2=curve_exe_js2[extension_start+idx_delay:-extension_end+idx_delay]-curve_js2_d
		error2=error2
		error2_flip=np.flipud(error2)
		###calcualte agumented input
		curve_cmd_js2_aug=curve_cmd_js2+error2_flip*gradient_step

		################################traverse curve for both arms#####################################
		curve_cmd_js1_aug_ext=clip_joints(robot1,et1.add_extension_egm_js(curve_cmd_js1_aug,extension_start=extension_start,extension_end=extension_end))
		curve_cmd_js2_aug_ext=clip_joints(robot2,et1.add_extension_egm_js(curve_cmd_js2_aug,extension_start=extension_start,extension_end=extension_end))

		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_aug_ext[0])
		_,curve_exe_js1_aug=et1.traverse_curve_js(curve_cmd_js1_aug_ext)

		et2.jog_joint(curve_cmd_js2_aug_ext[0])
		_,curve_exe_js2_aug=et2.traverse_curve_js(curve_cmd_js2_aug_ext)


		###get new error
		delta1_new=curve_exe_js1_aug[extension_start+idx_delay:-extension_end+idx_delay]-curve_exe_js1[extension_start+idx_delay:-extension_end+idx_delay]
		grad1=np.flipud(delta1_new)*gradient_step
		delta2_new=curve_exe_js2_aug[extension_start+idx_delay:-extension_end+idx_delay]-curve_exe_js2[extension_start+idx_delay:-extension_end+idx_delay]
		grad2=np.flipud(delta2_new)*gradient_step

		alpha=alpha_default
		skip=False
		#########################################adaptive step size######################
		for x in range(6):
			curve_cmd_js1_temp=clip_joints(robot1,curve_cmd_js1-alpha*grad1)
			curve_cmd_js2_temp=clip_joints(robot2,curve_cmd_js2-alpha*grad2)

			curve_cmd_js1_temp_ext=et1.add_extension_egm_js(curve_cmd_js1_temp,extension_start=extension_start,extension_end=extension_end)
			curve_cmd_js2_temp_ext=et1.add_extension_egm_js(curve_cmd_js2_temp,extension_start=extension_start,extension_end=extension_end)

			###jog both arm to start pose
			et1.jog_joint(curve_cmd_js1_temp_ext[0])
			timestamp1_temp,curve_exe_js1_temp=et1.traverse_curve_js(curve_cmd_js1_temp_ext)

			et2.jog_joint(curve_cmd_js2_temp_ext[0])
			timestamp2_temp,curve_exe_js2_temp=et2.traverse_curve_js(curve_cmd_js2_temp_ext)

			###calcualte error
			_,_,_,_,relative_path_exe_temp,relative_path_exe_R_temp=form_relative_path(curve_exe_js1_temp[extension_start+idx_delay:-extension_end+idx_delay],curve_exe_js2_temp[extension_start+idx_delay:-extension_end+idx_delay],robot1,robot2)

			lam_temp=calc_lam_cs(relative_path_exe_temp)
			speed_temp=np.gradient(lam_temp)/np.gradient(timestamp1_temp[extension_start+idx_delay:-extension_end+idx_delay])
			error_temp,angle_error_temp=calc_all_error_w_normal(relative_path_exe_temp,relative_path[:,:3],relative_path_exe_R_temp[:,:,-1],relative_path[:,3:])
			if np.max(error_temp)>np.max(error):
				alpha/=2
			else:
				skip=True
				break
		curve_cmd_js1=curve_cmd_js1_temp
		curve_cmd_js2=curve_cmd_js2_temp
		print('step size: ',alpha)
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