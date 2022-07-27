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

_lock=threading.Lock()
_streaming=False
curve_js2_exe=[]
timestamp2_exe=[]
egm2=None
num_EGM_points=0
def threadfunc():
	global _streaming, curve_js2_exe, timestamp2_exe, egm2, num_EGM_points
	curve_js2_exe=[]
	timestamp2_exe=[]
	while (_streaming or len(timestamp2_exe)<num_EGM_points):
		with _lock:
			try:
				res_i2, state_i2 = egm2.receive_from_robot()
				if res_i2:
					curve_js2_exe.append(np.radians(state_i2.joint_angles))
					timestamp2_exe.append(state_i2.robot_message.header.tm)
			except:
				traceback.print_exc()

def StartStreaming():
	global _streaming
	_streaming=True
	t=threading.Thread(target=threadfunc)
	t.start()
def StopStreaming():
	global _streaming
	_streaming=False

def main():
	global _streaming, curve_js2_exe, timestamp2_exe, egm2, num_EGM_points

	communication_delay=0.00
	data_dir='../../data/wood/'

	solution_dir='qp1/'
	
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

	vd=500
	idx=et1.downsample24ms(relative_path,vd)
	extension_start=100
	extension_end=100

	
	###read in reference points
	curve_js1_d=read_csv('dual_arm/wood/EGM_arm1_d.csv',header=None).values
	curve_js2_d=read_csv('dual_arm/wood/EGM_arm2_d.csv',header=None).values
	###read in cmd points
	curve_cmd_js1_ext=read_csv('dual_arm/wood/EGM_arm1.csv',header=None).values
	curve_cmd_js2_ext=read_csv('dual_arm/wood/EGM_arm2.csv',header=None).values

	num_EGM_points=len(curve_js1_d)
	
	iteration=100

	for i in range(iteration):
		################################traverse curve for both arms#####################################

		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_ext[0])
		et2.jog_joint(curve_cmd_js2_ext[0])

		curve_exe_js1=[]
		timestamp1=[]
		print('traversing trajectory')
		try:
			et1.clear_queue()
			et2.clear_queue()
			StartStreaming()
			time.sleep(communication_delay)
			for m in range(len(curve_cmd_js1_ext)):
				while True:
					res_i1, state_i1 = egm1.receive_from_robot()
					if res_i1:
						###send synchronously
						send_res1 = egm1.send_to_robot(curve_cmd_js1_ext[m])
						#save joint angles
						curve_exe_js1.append(np.radians(state_i1.joint_angles))
						timestamp1.append(state_i1.robot_message.header.tm)

						send_res2 = egm2.send_to_robot(curve_cmd_js2_ext[m])
						break
		except KeyboardInterrupt:
			StopStreaming()
			raise
		StopStreaming()

		time.sleep(0.1)

		timestamp1=np.array(timestamp1)/1000
		timestamp2=np.array(copy.deepcopy(timestamp2_exe))/1000
		
		curve_exe_js1=np.array(curve_exe_js1)
		curve_exe_js2=np.array(copy.deepcopy(curve_js2_exe))[-num_EGM_points:]

		print(len(timestamp2))
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
		et2.jog_joint(curve_cmd_js2_aug[0])

		curve_exe_js1_aug=[]
		timestamp1=[]
		print('traversing trajectory')
		try:
			et1.clear_queue()
			et2.clear_queue()
			StartStreaming()
			time.sleep(communication_delay)
			for m in range(len(curve_cmd_js1_aug)):
				while True:
					res_i1, state_i1 = egm1.receive_from_robot()
					if res_i1:
						###send synchronously
						send_res1 = egm1.send_to_robot(curve_cmd_js1_aug[m])
						#save joint angles
						curve_exe_js1_aug.append(np.radians(state_i1.joint_angles))
						timestamp1.append(state_i1.robot_message.header.tm)

						send_res2 = egm2.send_to_robot(curve_cmd_js2_aug[m])
						break
		except KeyboardInterrupt:
			StopStreaming()
			raise
		StopStreaming()

		time.sleep(0.1)

		timestamp1=np.array(timestamp1)/1000
		timestamp2=np.array(copy.deepcopy(timestamp2_exe))/1000
		
		curve_exe_js1_aug=np.array(curve_exe_js1_aug)
		curve_exe_js2_aug=np.array(copy.deepcopy(curve_js2_exe))[-num_EGM_points:]
		print(len(timestamp2))
		###get new error
		delta1_new=curve_exe_js1_aug-curve_exe_js1
		grad1=np.flipud(delta1_new)*gradient_step

		alpha=0.5
		curve_cmd_js1_ext=clip_joints(robot1,curve_cmd_js1_ext-alpha*grad1)

		delta2_new=curve_exe_js2_aug-curve_exe_js2
		grad2=np.flipud(delta2_new)*gradient_step

		alpha=0.5
		curve_cmd_js2_ext=clip_joints(robot2,curve_cmd_js2_ext-alpha*grad2)

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