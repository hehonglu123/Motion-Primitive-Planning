import sys
sys.path.append('../../toolbox')
from robot_def import *
sys.path.append('../')
from constraint_solver import *

def calc_js_from_theta(curve,curve_normal,theta_all,robot,q_init):
	curve_js=[]
	for i in range(len(curve)):
		if i==0:
			R_temp=direction2R(curve_normal[i],curve[i]-curve[i+1])
			q_seed=q_init
		else:
			R_temp=direction2R(curve_normal[i],curve[i-1]-curve[i])
			q_seed=curve_js[-1]
		R=np.dot(R_temp,Rz(theta_all[i]))
		q=robot.inv(curve[i],R,q_seed)[0]
		curve_js.append(q)
	return np.array(curve_js)

def calc_theta_from_js(curve,curve_js,robot):
	###calc theta at all points
	theta_all=[]
	for i in range(len(curve)):
		pose=robot.fwd(curve_js[i])
		if i==0:
			vec=curve[i]-curve[i+1]
		else:
			vec=curve[i-1]-curve[i]
		vec/=np.linalg.norm(vec)
		theta=get_angle2(vec,pose.R[:,0],pose.R[:,-1])
		theta_all.append(theta)

	return theta_all

def main():
	dataset='curve_2/'
	data_dir='../../data/'+dataset+'/'
	solution_dir='baseline/'
	curve = read_csv(data_dir+solution_dir+"Curve_in_base_frame.csv",header=None).values
	curve_js = read_csv(data_dir+solution_dir+"Curve_js.csv",header=None).values
	q_init=curve_js[0]


	robot=robot_obj('ABB_6640_180_255','../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../config/acceleration/6640acc_new.pickle')


	opt=lambda_opt(curve[:,:3],curve[:,3:],robot1=robot,steps=500,v_cmd=1000)

	q_out=opt.single_arm_stepwise_optimize(q_init)
	q_out_new=opt.single_arm_stepwise_optimize3(q_init,lamdot_des=500)

	lamdot_boundary=lambdadot_qlambda(robot,q_out,opt.lam)
	lamdot_boundary_new=lambdadot_qlambda(robot,q_out_new,opt.lam)

	speed=traj_speed_est(opt.robot1,q_out,opt.lam,opt.v_cmd)
	speed_new=traj_speed_est(opt.robot1,q_out_new,opt.lam,opt.v_cmd)

	plt.plot(opt.lam,lamdot_boundary,label=r'$\dot{\lambda}$ boundary old')
	plt.plot(opt.lam,lamdot_boundary_new,label=r'$\dot{\lambda}$ boundary new')

	plt.plot(opt.lam,speed,label=r'speed estimation old')
	plt.plot(opt.lam,speed_new,label=r'speed estimation new')

	plt.legend()
	# plt.ylim([0,3000])
	plt.xlabel(r'$\lambda$ (mm)')
	plt.ylabel(r'$\dot{\lambda}$ (mm/s)')
	plt.title(r'$\dot{\lambda}$ Boundary Profile' )
	plt.show()

if __name__ == "__main__":
	main()