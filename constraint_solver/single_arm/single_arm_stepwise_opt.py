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


	# q_init=[0.085918203,	0.096852813,	0.284197147,	2.563882607,	-1.344704035,	-3.032035596]

	robot=robot_obj('ABB_6640_180_255','../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../config/acceleration/6640acc_new.pickle')


	opt=lambda_opt(curve[:,:3],curve[:,3:],robot1=robot,steps=500,v_cmd=500)

	q_out=opt.single_arm_stepwise_optimize(q_init)

	####output to trajectory csv
	np.savetxt('trajectory/stepwise_opt/arm1.csv',q_out,delimiter=',')


	# speed=calc_lamdot(q_out,opt.lam,opt.robot1,1)
	speed=traj_speed_est(opt.robot1,q_out,opt.lam,opt.v_cmd)


	plt.plot(opt.lam,speed,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([0,1.2*opt.v_cmd])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/stepwise_opt/results.png")
	plt.clf()
	# plt.show()

	# ###calc theta at all points
	# theta_all=calc_theta_from_js(opt.curve,q_out,robot)
	
	# dtheta_all=np.random.uniform(low=-0.1,high=0.1,size=len(theta_all))
	# dv_all=[]
	# dtheta_all=[]
	
	# for k in range(10):
	# 	dtheta=np.random.uniform(low=-0.01,high=0.01,size=len(theta_all))
	# 	theta_all_temp=copy.deepcopy(theta_all)
	# 	theta_all_temp+=dtheta
	# 	curve_js_reform=calc_js_from_theta(opt.curve,opt.curve_normal,theta_all_temp,robot,q_out[0])
	# 	speed_temp=traj_speed_est(opt.robot1,curve_js_reform,opt.lam,opt.v_cmd)
	# 	dv_all.append(speed_temp-speed)
	# 	dtheta_all.append(dtheta)

	# dtheta_all=np.array(dtheta_all)
	# dv_all=np.array(dv_all)

	# G=dv_all.T@np.linalg.pinv(dtheta_all.T)
	
	# plt.figure()
	# im=plt.imshow(G, cmap='hot', interpolation='nearest')
	# plt.colorbar(im)
	# plt.title("Numerical Gradient")
	# plt.xlabel('d_theta')
	# plt.ylabel('d_vmin')
	# plt.show()
if __name__ == "__main__":
	main()