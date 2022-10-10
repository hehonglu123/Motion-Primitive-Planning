import sys
sys.path.append('../')
from constraint_solver import *
from MotionSend import *
class Geeks:
    pass

def main():

	data_dir='../../data/wood/'
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	v_cmd=1500

	with open(data_dir+'dual_arm/abb1200_2.yaml') as file:
	    H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	base2_R=H_1200[:3,:3]
	base2_p=1000*H_1200[:-1,-1]

	base2_k,base2_theta=R2rot(base2_R)

	with open(data_dir+'dual_arm/tcp.yaml') as file:
	    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	robot1=abb6640(d=50, acc_dict_path='../../toolbox/robot_info/6640acc_new.pickle')
	robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1], acc_dict_path='../../toolbox/robot_info/1200acc_new.pickle')

	ms = MotionSend(robot2=robot2,base2_R=base2_R,base2_p=base2_p)

	#read in initial curve pose
	with open(data_dir+'baseline/curve_pose.yaml') as file:
		blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	# curve_js1=read_csv(data_dir+"Curve_js.csv",header=None).values
	# q_init1=curve_js1[0]
	# q_init2=ms.calc_robot2_q_from_blade_pose(blade_pose,base2_R,base2_p)
	q_init2=np.zeros(6)
	
	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p,steps=500,v_cmd=v_cmd)

	###########################################diff evo opt############################################
	##x:q_init2,base2_x,base2_y,base2_theta,theta_0
	lower_limit=np.hstack((robot2.lower_limit,[0,0],[-np.pi],[-np.pi]))
	upper_limit=np.hstack((robot2.upper_limit,[3000,3000],[np.pi],[np.pi]))
	bnds=tuple(zip(lower_limit,upper_limit))
	res = differential_evolution(opt.dual_arm_opt_w_pose_3dof, bnds, args=None,workers=-1,
									x0 = np.hstack((q_init2,base2_p[0],base2_p[1],base2_theta,[0])),
									strategy='best1bin', maxiter=700,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	print(res)

	# res=Geeks()
	# res.x=np.array([ 9.65670201e-01, -5.05316283e-01,  5.69817112e-01, -2.64359959e+00,
 #       -1.07833047e+00, -3.83256160e+00,  2.68322188e+03,  8.62483843e+01,
 #        9.78732092e-01,  1.64867478e+00])
	# print(opt.dual_arm_opt_w_pose_3dof(res.x))

	q_init2=res.x[:6]
	base2_p=np.array([res.x[6],res.x[7],790.5])		###fixed z height
	base2_theta=res.x[8]
	base2_R=Rz(base2_theta)


	pose2_world_now=robot2.fwd(q_init2,base2_R,base2_p)


	R_temp=direction2R(pose2_world_now.R@opt.curve_normal[0],-opt.curve[1]+opt.curve[0])
	R=np.dot(R_temp,Rz(res.x[-1]))

	q_init1=robot1.inv(pose2_world_now.p,R)[0]

	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p,steps=50000)
	q_out1,q_out2=opt.dual_arm_stepwise_optimize(q_init1,q_init2,w1=0.02,w2=0.01,base2_R=base2_R,base2_p=base2_p)

	####output to trajectory csv
	df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
	df.to_csv('trajectory/arm1.csv',header=False,index=False)
	df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
	df.to_csv('trajectory/arm2.csv',header=False,index=False)

	###output to pose yaml
	H=np.eye(4)
	H[:-1,-1]=base2_p/1000.
	H[:3,:3]=base2_R

	with open(r'trajectory/abb1200.yaml', 'w') as file:
		documents = yaml.dump({'H':H.tolist()}, file)

	###dual lambda_dot calc
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