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
timestamp2=[]
egm2=None
def threadfunc():
	global _streaming, curve_js2_exe, timestamp2, egm2
	while(_streaming):
		with _lock:
			try:
				res_i2, state_i2 = egm2.receive_from_robot()
				if res_i2:
					curve_js2_exe.append(np.radians(state_i2.joint_angles))
					timestamp2.append(state_i2.robot_message.header.tm)
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
	global _streaming, curve_js2_exe, timestamp2, egm2

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
	iteration=30
	adjust_weigt_it=10
	for i in range(iteration):

		##add extension
		curve_cmd_js1_ext=et1.add_extension_egm_js(curve_cmd_js1,extension_num=extension_num)
		curve_cmd_js2_ext=et1.add_extension_egm_js(curve_cmd_js2,extension_num=extension_num)


		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_ext[0])
		et2.jog_joint(curve_cmd_js2_ext[0])

		################################traverse curve for both arms#####################################

		curve_exe_js1=[]
		timestamp1=[]
		curve_exe_js2=[]
		print('traversing trajectory')
		try:
			StartStreaming()
			for i in range(len(curve_cmd_js2_ext)):
				while True:
					res_i1, state_i1 = egm1.receive_from_robot()
					if res_i1:
						###send synchronously
						send_res1 = egm1.send_to_robot(curve_cmd_js1_ext[i])
						#save joint angles
						curve_exe_js1.append(np.radians(state_i1.joint_angles))
						timestamp1.append(state_i1.robot_message.header.tm)

						send_res2 = egm2.send_to_robot(curve_cmd_js2_ext[i])
						break
		except KeyboardInterrupt:
			raise
		StopStreaming()
		time.sleep(1)

		timestamp1=np.array(timestamp1)/1000
		timestamp2=np.array(timestamp2)/1000

		print(len(timestamp1))
		print(len(timestamp2))
		plt.plot(timestamp1)
		plt.show()
		print(timestamp1[-1]-timestamp1[0])
		print(timestamp2[-1]-timestamp2[0])

		curve_exe_js1=np.array(curve_exe_js1)
		curve_exe_js2=np.array(copy.deepcopy(curve_js2_exe))
		##############################calcualte error########################################
		# relative_path_exe,relative_path_exe_R=form_relative_path(robot1,robot2,curve_exe_js1,curve_exe_js2,base2_R,base2_p)
		# lam=calc_lam_cs(relative_path_exe)
		# speed=np.gradient(lam)/np.gradient(timestamp1)
		# error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])
		# print(max(error))


		##############################ILC########################################
		error1=curve_exe_js1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_js1_d
		error1_flip=np.flipud(error1)
		# print(error1)
		###calcualte agumented input
		curve_cmd_js1_aug=curve_cmd_js1+error1_flip

		error2=curve_exe_js2[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_js2_d
		error2_flip=np.flipud(error2)
		###calcualte agumented input
		curve_cmd_js2_aug=curve_cmd_js2+error2_flip

		##add extension
		curve_cmd_js1_ext_aug=et1.add_extension_egm_js(curve_cmd_js1_aug,extension_num=extension_num)
		curve_cmd_js2_ext_aug=et1.add_extension_egm_js(curve_cmd_js2_aug,extension_num=extension_num)


		###jog both arm to start pose
		et1.jog_joint(curve_cmd_js1_ext_aug[0])
		et2.jog_joint(curve_cmd_js2_ext_aug[0])

		################################traverse curve for both arms#####################################

		curve_exe_js1_aug=[]
		curve_exe_js2_aug=[]
		print('traversing trajectory')
		try:
			for i in range(len(curve_cmd_js2_ext)):
				while True:
					res_i1, state_i1 = egm1.receive_from_robot()
					if res_i1:
				
						send_res1 = egm1.send_to_robot(curve_cmd_js1_ext_aug[i])
						#save joint angles
						curve_exe_js1_aug.append(np.radians(state_i1.joint_angles))
						

						send_res2 = egm2.send_to_robot(curve_cmd_js2_ext_aug[i])
						break
		except KeyboardInterrupt:
			raise
		time.sleep(1)

		curve_exe_js1_aug=np.array(curve_exe_js1_aug)
		curve_exe_js2_aug=np.array(curve_exe_js2_aug)

		###get new error
		delta1_new=curve_exe_js1_aug[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_exe_js1[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]

		grad1=np.flipud(delta1_new)

		alpha=0.5
		curve_cmd_js1=curve_cmd_js1-alpha*grad1

		delta2_new=curve_exe_js2_aug[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]-curve_exe_js2[extension_num+et1.idx_delay:-extension_num+et1.idx_delay]

		grad2=np.flipud(delta2_new)

		alpha=0.5
		curve_cmd_js2=curve_cmd_js2-alpha*grad1

	##############################plot error#####################################

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(lam, speed, 'g-', label='Speed')
	ax2.plot(lam, error, 'b-',label='Error')
	ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
	ax2.axis(ymin=0,ymax=100)

	ax1.set_xlabel('lambda (mm)')
	ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
	ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
	plt.title("Speed and Error Plot")
	ax1.legend(loc=0)

	ax2.legend(loc=0)

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()