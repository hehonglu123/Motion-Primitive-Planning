import sys
sys.path.append('../')
from constraint_solver import *
from MotionSend import *


def main():

	data_dir='../../data/wood/'
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	v_cmd=1100

	with open(data_dir+'dual_arm/abb1200.yaml') as file:
	    H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	with open(data_dir+'dual_arm/tcp.yaml') as file:
	    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	robot1=abb6640(d=50, acc_dict_path='../../toolbox/robot_info/6640acc.pickle')
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1], acc_dict_path='../../toolbox/robot_info/1200acc.pickle')

	ms = MotionSend(robot2=robot2,base2_R=base2_R,base2_p=base2_p)

	#read in initial curve pose
	with open(data_dir+'blade_pose.yaml') as file:
		blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	curve_js1=read_csv(data_dir+"Curve_js.csv",header=None).values
	q_init1=curve_js1[0]
	q_init2=ms.calc_robot2_q_from_blade_pose(blade_pose,base2_R,base2_p)
	
	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p,steps=500,v_cmd=v_cmd)

	###########################################diff evo opt############################################
	lower_limit=np.append(robot2.lower_limit,[-np.pi])
	upper_limit=np.append(robot2.upper_limit,[np.pi])
	bnds=tuple(zip(lower_limit,upper_limit))
	res = differential_evolution(opt.dual_arm_init_opt, bnds, args=None,workers=-1,
									x0 = np.append(q_init2,[0]),
									strategy='best1bin', maxiter=1,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	print(res)


	q_init2=res.x[:-1]
	pose2_world_now=robot2.fwd(q_init2,opt.base2_R,opt.base2_p)

	R_temp=direction2R(np.dot(pose2_world_now.R,opt.curve_normal[0]),-opt.curve[1]+opt.curve[0])
	R=np.dot(R_temp,Rz(res.x[-1]))

	q_init1=robot1.inv(pose2_world_now.p,R)[0]
	###########################################stepwise qp solver#####################################################
	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p,steps=50000)
	q_out1, q_out2=opt.dual_arm_stepwise_optimize(q_init1,q_init2,w1=0.02,w2=0.01)

	####output to trajectory csv
	df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
	df.to_csv('trajectory/arm1.csv',header=False,index=False)
	df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
	df.to_csv('trajectory/arm2.csv',header=False,index=False)

	###dual lambda_dot calc
	# dlam_out=calc_lamdot(np.hstack((q_out1,q_out2)),opt.lam[:len(q_out2)],np.tile(opt.joint_vel_limit,2),1)
	speed,speed1,speed2=traj_speed_est_dual(robot1,robot2,q_out1[::100],q_out2[::100],opt.base2_R,opt.base2_p,opt.lam[::100],v_cmd)

	print('speed min: ', min(speed))

	plt.plot(opt.lam[::100],speed,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("DUALARM max lambda_dot vs lambda (path index)")
	plt.ylim([0,1.2*v_cmd])
	plt.savefig("trajectory/results.png")
	# plt.show()


if __name__ == "__main__":
	main()