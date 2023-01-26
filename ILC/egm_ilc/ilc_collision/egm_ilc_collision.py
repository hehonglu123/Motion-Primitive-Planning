import sys
sys.path.append('../../toolbox/egm_toolbox')

from robots_def import *
from error_check import *
from lambda_calc import *
from EGM_toolbox import *

from ilc_env import *

def barrier(x):
	a=20;b=5;e=0.1;l=1;
	return np.divide(a*b*(x-e),l+b*(x-e))

def collision_free_gen(robot,ilc_env,curve_js,d_safe=0.01):
	curve_js_out=[]
	count=0
	for q in curve_js:
		min_d, J2C, d=ilc_env.check_collision_single('ABB_6640_180_255','ball_link',q)
		###create copy instance
		q_out=copy.deepcopy(q)
		###loop until collision free
		count+=1
		while min_d<d_safe:
			d_norm=d/np.linalg.norm(d)
			Jacobian2C=robot.jacobian(q)[:,:J2C+1]

			if min_d<0:
				vd=(-min_d+d_safe+0.1)*d_norm
			else:
				vd=(-min_d+d_safe+0.1)*-d_norm

			dq=np.hstack(((np.linalg.pinv(Jacobian2C)[:,3:6]@vd).flatten(),np.zeros(max(len(q)-J2C-1,0))))
			q_out+=dq
			# print(min_d,count)
			
			min_d, J2C, d=ilc_env.check_collision_single('ABB_6640_180_255','ball_link',q_out)

		curve_js_out.append(q_out)

	return np.array(curve_js_out)

def collision_free_gen_qp(robot,ilc_env,curve_js,d_safe=0.01):
	curve_js_out=[]
	count=0
	Kq=.01*np.eye(n)    #small value to make sure positive definite

	for q in curve_js:
		min_d, J2C, d=ilc_env.check_collision_single('ABB_6640_180_255','ball_link',q)
		###create copy instance
		q_out=copy.deepcopy(q)
		###loop until collision free
		count+=1
		while min_d<d_safe:
			d_norm=d/np.linalg.norm(d)
			Jacobian2C=robot.jacobian(q)[:,:J2C+1]

			if min_d<0:
				vd=(-min_d+d_safe+0.1)*d_norm
			else:
				vd=(-min_d+d_safe+0.1)*-d_norm

			Jp=Jacobian2C[:,3:6]
			vd=-np.dot(Kp,EP)
			H=np.dot(np.transpose(Jp),Jp)+Kq
			H=(H+np.transpose(H))/2
			f=-np.dot(np.transpose(Jp),vd)
			dq=solve_qp(H,f)

			q_out+=dq
			
			min_d, J2C, d=ilc_env.check_collision_single('ABB_6640_180_255','ball_link',q_out)

		curve_js_out.append(q_out)

	return np.array(curve_js_out)



def traverse_curve_js_qp(robot,et,curve_js,ilc_env):
	et.clear_queue()
	curve_exe_js=[]
	timestamp=[]
	###traverse curve
	print('traversing trajectory')
	try:
		for i in range(len(curve_js)):
			while True:
				res_i, state_i = et.egm.receive_from_robot()
				if res_i:
					#save joint angles
					q_cur=np.radians(state_i.joint_angles)
					curve_exe_js.append(q_cur)
					timestamp.append(state_i.robot_message.header.tm)
					###########################collision checking#####################################
					min_d, J2C, d=ilc_env.check_collision_single('ABB_6640_180_255','ball_link',q_cur)

					if min_d<0.1:
						d_norm=d/np.linalg.norm(d)
						push_coeff=barrier(min_d)
						print(push_coeff)
						d_push=np.sign(min_d)*push_coeff*d_norm			###m to mm, calculate push vector
						Jacobian2C=robot.jacobian(q_cur)[:,:J2C+1]
						d_q=np.dot(np.linalg.pinv(Jacobian2C[3:]),d_push)
						d_q=np.append(d_q,np.zeros(len(curve_js[0])-len(d_q)))		###reshape to full 6
						print(d_q)
						q_cmd=q_cur+d_q+curve_js[i+1]-curve_js[i]
					else:
						q_cmd=curve_js[i]

					send_res = et.egm.send_to_robot(q_cmd)
					break
	except KeyboardInterrupt:
		raise
	
	timestamp=np.array(timestamp)/1000
	# plt.plot(timestamp)
	# plt.show()

	return timestamp,np.array(curve_exe_js)

def clip_joints(robot,curve_js):
	curve_js_out=[]
	for q in curve_js:
		curve_js_out.append(np.clip(q,robot.lower_limit,robot.upper_limit))

	return np.array(curve_js_out)

def egm_ilc(robot,curve_js_d,curve_js_cmd,ilc_env):
	egm = rpi_abb_irc5.EGM()
	et=EGM_toolbox(egm,robot)
	
	for i in range(10):
		###jog both arm to start pose
		et.jog_joint(curve_js_cmd[0])
		
		###traverse the curve
		timestamp,curve_exe_js=et.traverse_curve_js(curve_js_cmd)
		###############################ILC########################################
		error=curve_exe_js-curve_js_d
		error_flip=np.flipud(error)
		###calcualte agumented input
		curve_js_cmd_aug=clip_joints(robot,curve_js_cmd+error_flip)

		time.sleep(1)

		###move to start first
		print('moving to start point')
		et.jog_joint(curve_js_cmd_aug[0])
		###traverse the curve
		timestamp_aug,curve_exe_js_aug=et.traverse_curve_js(curve_js_cmd_aug)

		###get new error
		delta_new=curve_exe_js_aug-curve_exe_js

		grad=np.flipud(delta_new)

		alpha=0.2
		curve_js_cmd=curve_js_cmd-alpha*grad

		###collision free path
		curve_js_cmd=clip_joints(robot,collision_free_gen(robot,ilc_env,curve_js_cmd))

		np.savetxt('curve_js_cmd.csv',curve_js_cmd,delimiter=',')


def ilc_collision(robot,curve_js_d,curve_js_cmd,ilc_env):

	for i in range(10):
		###############################ILC########################################
		error=curve_js_cmd-curve_js_d
		error_flip=np.flipud(error)
		###calcualte agumented input
		curve_js_cmd_aug=clip_joints(robot,curve_js_cmd+error_flip)

		###get new error
		delta_new=curve_js_cmd_aug-curve_js_cmd

		grad=np.flipud(delta_new)

		alpha=0.2
		curve_js_cmd=curve_js_cmd-alpha*grad

		###collision free path
		curve_js_cmd=clip_joints(robot,collision_free_gen(robot,ilc_env,curve_js_cmd))

		np.savetxt('curve_js_cmd.csv',curve_js_cmd,delimiter=',')


def main():
	q_init=np.array([np.pi/6,0,0,0,0,0])
	q_end=np.array([-np.pi/6,0,0,0,0,0])
	q_mid=np.array([0,-np.pi/6,0,0,0,0])

	num_points=1000

	curve_js_d=np.linspace(q_init,q_end,num_points)		###desired joint trajectory

	robot=robot_obj('ABB_6640_180_255','../../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50)

	ilc_env=ILC_Tess_Env('../../../config/urdf/')	

	curve_js_out=collision_free_gen(robot,ilc_env,curve_js)

	np.savetxt('collision_free.csv',curve_js_out,delimiter=',')

	# egm = rpi_abb_irc5.EGM()
	# et=EGM_toolbox(egm,robot)
	# et.jog_joint(q_init)
	# traverse_curve_js_qp(robot,et,curve_js,ilc_env)



def exec_ilc():
	robot=robot_obj('ABB_6640_180_255','../../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50)
	egm = rpi_abb_irc5.EGM()
	et=EGM_toolbox(egm,robot)
	curve_js=np.loadtxt('collision_free.csv',delimiter=',')
	et.jog_joint(curve_js[0])
	et.traverse_curve_js(curve_js)

def exec_ilc_collision():
	q_init=np.array([np.pi/6,0,0,0,0,0])
	q_end=np.array([-np.pi/6,0,0,0,0,0])
	q_mid=np.array([0,-np.pi/6,0,0,0,0])

	num_points=1000

	curve_js_d=np.linspace(q_init,q_end,num_points)		###desired joint trajectory

	robot=robot_obj('ABB_6640_180_255','../../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../../config/paintgun.csv',d=50)

	ilc_env=ILC_Tess_Env('../../../config/urdf/')

	curve_js_cmd=np.vstack((np.linspace(q_init,q_mid,int(num_points/2)),np.linspace(q_mid,q_end,int(num_points/2))))

	ilc_collision(robot,curve_js_d,curve_js_cmd,ilc_env)

if __name__ == '__main__':
	# main()
	# exec_ilc()
	exec_ilc_collision()






	