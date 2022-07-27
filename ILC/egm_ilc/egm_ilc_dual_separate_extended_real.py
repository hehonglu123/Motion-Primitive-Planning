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
	dataset='from_NX/'
	data_dir='../../data/'+dataset
	solution_dir='qp5/'
	
	robot1=abb6640(d=50)
	with open(data_dir+'dual_arm/'+solution_dir+'abb1200.yaml') as file:
		H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	with open(data_dir+'dual_arm/'+solution_dir+'tcp.yaml') as file:
		H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

	egm1 = rpi_abb_irc5.EGM(port=6510)
	egm2 = rpi_abb_irc5.EGM(port=6511)

	et1=EGM_toolbox(egm1,robot1)
	et2=EGM_toolbox(egm2,robot2)

	curve_js1=read_csv(data_dir+'dual_arm/'+solution_dir+'arm1.csv',header=None).values
	curve_js2=read_csv(data_dir+'dual_arm/'+solution_dir+'arm2.csv',header=None).values
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	vd=1500
	idx=et1.downsample24ms(relative_path,vd)
	extension_start=100
	extension_end=100

	###read in reference points
	curve_js1_d=read_csv('dual_arm/'+dataset+'EGM_arm1_d.csv',header=None).values
	curve_js2_d=read_csv('dual_arm/'+dataset+'EGM_arm2_d.csv',header=None).values
	###read in cmd points
	curve_cmd_js1_ext=read_csv('dual_arm/'+dataset+'EGM_arm1.csv',header=None).values
	curve_cmd_js2_ext=read_csv('dual_arm/'+dataset+'EGM_arm2.csv',header=None).values
	
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
			relative_path_exe,relative_path_exe_R=form_relative_path(robot1,robot2,curve_exe_js1[extension_start:-extension_end],curve_exe_js2[extension_start:-extension_end],base2_R,base2_p)

			lam=calc_lam_cs(relative_path_exe)
			speed=np.gradient(lam)/np.gradient(timestamp1[extension_start:-extension_end])
			error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
		
		print(max(error))


		##############################ILC########################################
		gradient_step=1
		error1=curve_exe_js1-curve_js1_d
		error1=error1
		error1_flip=np.flipud(error1)
		# print(error1)
		###calcualte agumented input
		curve_cmd_js1_aug=clip_joints(robot1,curve_cmd_js1_ext+error1_flip*gradient_step)

		error2=curve_exe_js2-curve_js2_d
		error2=error2
		error2_flip=np.flipud(error2)
		###calcualte agumented input
		curve_cmd_js2_aug=clip_joints(robot2,curve_cmd_js2_ext+error2_flip*gradient_step)

		################################traverse curve for both arms#####################################

		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_aug[0])
		_,curve_exe_js1_aug=et1.traverse_curve_js(curve_cmd_js1_aug)

		et2.jog_joint(curve_cmd_js2_aug[0])
		_,curve_exe_js2_aug=et2.traverse_curve_js(curve_cmd_js2_aug)




		###get new error
		delta1_new=curve_exe_js1_aug-curve_exe_js1
		grad1=np.flipud(delta1_new)*gradient_step
		delta2_new=curve_exe_js2_aug-curve_exe_js2
		grad2=np.flipud(delta2_new)*gradient_step

		alpha=alpha_default
		skip=False
		#########################################adaptive step size######################
		for x in range(6):
			curve_cmd_js1_ext_temp=clip_joints(robot1,curve_cmd_js1_ext-alpha*grad1)
			curve_cmd_js2_ext_temp=clip_joints(robot2,curve_cmd_js2_ext-alpha*grad2)

			###jog both arm to start pose
			et1.jog_joint(curve_cmd_js1_ext_temp[0])
			timestamp1_temp,curve_exe_js1_temp=et1.traverse_curve_js(curve_cmd_js1_ext_temp)

			et2.jog_joint(curve_cmd_js2_ext_temp[0])
			timestamp2_temp,curve_exe_js2_temp=et2.traverse_curve_js(curve_cmd_js2_ext_temp)

			###calcualte error
			relative_path_exe_temp,relative_path_exe_R_temp=form_relative_path(robot1,robot2,curve_exe_js1_temp[extension_start:-extension_end],curve_exe_js2_temp[extension_start:-extension_end],base2_R,base2_p)

			lam_temp=calc_lam_cs(relative_path_exe_temp)
			speed_temp=np.gradient(lam_temp)/np.gradient(timestamp1_temp[extension_start:-extension_end])
			error_temp,angle_error_temp=calc_all_error_w_normal(relative_path_exe_temp,relative_path[:,:3],relative_path_exe_R_temp[:,:,-1],relative_path[:,3:])
			if np.max(error_temp)>np.max(error):
				alpha/=2
			else:
				skip=True
				break
		curve_cmd_js1_ext=curve_cmd_js1_ext_temp
		curve_cmd_js2_ext=curve_cmd_js2_ext_temp
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
		ax1.legend(loc=0)

		ax2.legend(loc=0)

		plt.legend()
		# plt.show()
		plt.savefig('iteration '+str(i))
		plt.clf()

		plt.plot(error1,label=['joint1','joint2','joint3','joint4','joint5','joint6'])
		plt.legend()
		plt.savefig('iteration '+str(i)+'_r1')
		plt.clf()

		plt.plot(error2,label=['joint1','joint2','joint3','joint4','joint5','joint6'])
		plt.legend()
		plt.savefig('iteration '+str(i)+'_r2')
		plt.clf()

		###save EGM commands
		df=DataFrame({'q0':curve_cmd_js1_ext[:,0],'q1':curve_cmd_js1_ext[:,1],'q2':curve_cmd_js1_ext[:,2],'q3':curve_cmd_js1_ext[:,3],'q4':curve_cmd_js1_ext[:,4],'q5':curve_cmd_js1_ext[:,5]})
		df.to_csv('EGM_arm1.csv',header=False,index=False)
		df=DataFrame({'q0':curve_js1_d[:,0],'q1':curve_js1_d[:,1],'q2':curve_js1_d[:,2],'q3':curve_js1_d[:,3],'q4':curve_js1_d[:,4],'q5':curve_js1_d[:,5]})
		df.to_csv('EGM_arm1_d.csv',header=False,index=False)
		df=DataFrame({'q0':curve_cmd_js2_ext[:,0],'q1':curve_cmd_js2_ext[:,1],'q2':curve_cmd_js2_ext[:,2],'q3':curve_cmd_js2_ext[:,3],'q4':curve_cmd_js2_ext[:,4],'q5':curve_cmd_js2_ext[:,5]})
		df.to_csv('EGM_arm2.csv',header=False,index=False)
		df=DataFrame({'q0':curve_js2_d[:,0],'q1':curve_js2_d[:,1],'q2':curve_js2_d[:,2],'q3':curve_js2_d[:,3],'q4':curve_js2_d[:,4],'q5':curve_js2_d[:,5]})
		df.to_csv('EGM_arm2_d.csv',header=False,index=False)

if __name__ == '__main__':
	main()