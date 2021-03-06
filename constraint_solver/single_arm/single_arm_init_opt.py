import sys
sys.path.append('../')
from constraint_solver import *


def main():
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("../../train_data/from_ge/Curve_in_base_frame2.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	robot=abb6640(d=50)
	opt=lambda_opt(curve,curve_normal,robot1=robot)

	lowerer_limit=[-np.pi]
	upper_limit=[np.pi]
	bnds=tuple(zip(lowerer_limit,upper_limit))
	res = differential_evolution(opt.single_arm_theta0_opt, bnds, args=None,workers=1,
									x0 = [0],
									strategy='best1bin', maxiter=500,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	


	print(res)

	R_temp=opt.direction2R(opt.curve_normal[0],-curve[1]+curve[0])
	R=np.dot(R_temp,Rz(res.x[0]))
	q_init=opt.robot1.inv(curve[0],R)[0]
	q_out=opt.single_arm_stepwise_optimize(q_init)

	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/init_opt/arm1.csv',header=False,index=False)

	dlam_out=calc_lamdot(q_out,opt.lam[:len(q_out)],opt.robot1.joint_vel_limit,1)


	plt.plot(opt.lam[:len(q_out)-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([500,2000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/init_opt/results.png")
	plt.show()
	

if __name__ == "__main__":
	main()