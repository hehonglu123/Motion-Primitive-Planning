import sys
sys.path.append('../')
from constraint_solver import *
from MotionSend import *
class Geeks:
	pass

def main():

	data_dir='../../data/curve_1/'
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	v_cmd=1333

	H_1200=np.loadtxt(data_dir+'dual_arm/abb1200.csv',delimiter=',')

	base2_R=H_1200[:3,:3]
	base2_p=H_1200[:-1,-1]

	base2_k,base2_theta=R2rot(base2_R)

	robot1=robot_obj('ABB_6640_180_255','../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../toolbox/robot_info/6640acc_new.pickle')
	robot2=robot_obj('ABB_1200_5_90','../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=data_dir+'dual_arm/tcp.csv',acc_dict_path='../../toolbox/robot_info/1200acc_new.pickle')

	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=500,v_cmd=v_cmd)

	###########################################diff evo opt############################################
	##x:q_init2,base2_x,base2_y,base2_theta,theta_0
	lower_limit=np.hstack((robot2.lower_limit,[0,0],[-np.pi],[-np.pi]))
	upper_limit=np.hstack((robot2.upper_limit,[3000,3000],[np.pi],[np.pi]))
	bnds=tuple(zip(lower_limit,upper_limit))
	res = differential_evolution(opt.dual_arm_opt_w_pose_3dof, bnds, args=None,workers=-1,
									x0 = np.hstack((np.zeros(6),base2_p[0],base2_p[1],base2_theta,[0])),
									strategy='best1bin', maxiter=1800,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=True,
									polish=True, init='latinhypercube',
									atol=0)
	print(res)

	# res=Geeks()
	# res.x=np.array([ 8.30204750e-01,  1.07473533e+00, -1.50759224e-01,  5.54649365e-02,
 #       -2.99114694e-01, -5.31835132e+00,  2.40987396e+03,  1.34790369e+03,
 #        2.61895317e+00,  2.71631470e+00])
	# print(opt.dual_arm_opt_w_pose_3dof(res.x))

	q_init2=res.x[:6]
	base2_p=np.array([res.x[6],res.x[7],765.5])		###fixed z height
	base2_theta=res.x[8]
	base2_R=Rz(base2_theta)

	robot2.base_H=H_from_RT(base2_R,base2_p)
	pose2_world_now=robot2.fwd(q_init2,world=True)


	R_temp=direction2R(pose2_world_now.R@opt.curve_normal[0],pose2_world_now.R@(-opt.curve[1]+opt.curve[0]))
	R=np.dot(R_temp,Rz(res.x[-1]))

	q_init1=robot1.inv(pose2_world_now.R@opt.curve[0]+pose2_world_now.p,R)[0]

	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=50000)
	q_out1,q_out2,_,_=opt.dual_arm_stepwise_optimize(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,w1=0.01,w2=0.025)

	####output to trajectory csv
	df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
	df.to_csv('trajectory/arm1.csv',header=False,index=False)
	df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
	df.to_csv('trajectory/arm2.csv',header=False,index=False)

	###output to pose yaml
	H=np.eye(4)
	H[:-1,-1]=base2_p
	H[:3,:3]=base2_R

	np.savetxt('trajectory/base.csv', H, delimiter=',')

	###dual lambda_dot calc
	speed,speed1,speed2=traj_speed_est_dual(robot1,robot2,q_out1[::100],q_out2[::100],opt.lam[::100],v_cmd)

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