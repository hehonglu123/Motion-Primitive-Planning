import sys
sys.path.append('../')
from constraint_solver import *
sys.path.append('../../toolbox')
from robots_def import *

def main():

	###read actual curve
	q_init2=[0.181847959,	0.435384629,	0.465584225,	0.677493313,	-1.070550513,	-0.266374094]

	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("curve_poses/relative_path_tool_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	relative_path=np.vstack((curve_x, curve_y, curve_z)).T
	relative_path_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	robot1=abb1200()
	robot2=abb6640()
	opt=lambda_opt(relative_path,relative_path_direction,robot1=robot1,robot2=robot2,base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]]),	base2_p=np.array([3000,0,0]))


	###########################################diff evo opt############################################
	lowerer_limit=np.append(robot2.lowerer_limit,[-np.pi])
	upper_limit=np.append(robot2.upper_limit,[np.pi])
	bnds=tuple(zip(lowerer_limit,upper_limit))
	res = differential_evolution(opt.dual_arm_opt, bnds, args=None,workers=-1,
									x0 = np.append(q_init2,[0]),
									strategy='best1bin', maxiter=500,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	print(res)

	q_init2=res.x[:-1]
	pose2_world_now=robot2.fwd(q_init2,opt.base2_R,opt.base2_p)

	R_temp=opt.direction2R(np.dot(pose2_world_now.R,opt.curve_normal[0]),-opt.curve[1]+opt.curve[0])
	R=np.dot(R_temp,Rz(res.x[-1]))

	q_init1=robot1.inv(pose2_world_now.p,R)[0]
	###########################################stepwise qp solver#####################################################
	q_out1, q_out2=opt.dual_arm_stepwise_optimize(q_init1,q_init2)

	####output to trajectory csv
	df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
	df.to_csv('trajectory/arm1.csv',header=False,index=False)
	df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
	df.to_csv('trajectory/arm2.csv',header=False,index=False)

	###dual lambda_dot calc
	dlam_out=calc_lamdot(np.hstack((q_out1,q_out2)),opt.lam[:len(q_out2)],np.tile(opt.joint_vel_limit,2),1)
	print('lamdadot min: ', min(dlam_out))

	plt.plot(opt.lam[:len(q_out2)-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("DUALARM max lambda_dot vs lambda (path index)")
	plt.ylim([0.5,3.5])
	plt.savefig("trajectory/results.png")
	plt.show()


if __name__ == "__main__":
	main()