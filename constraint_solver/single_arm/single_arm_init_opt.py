import sys
sys.path.append('../')
from constraint_solver import *


def main():
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = read_csv("curve_poses/curve_pose0/Curve_backproj_in_base_frame.csv", names=col_names)
	curve_x=data['X'].tolist()
	curve_y=data['Y'].tolist()
	curve_z=data['Z'].tolist()
	curve_direction_x=data['direction_x'].tolist()
	curve_direction_y=data['direction_y'].tolist()
	curve_direction_z=data['direction_z'].tolist()
	curve=np.vstack((curve_x, curve_y, curve_z)).T
	curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

	###read actual curve
	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("curve_poses/curve_pose0/Curve_backproj_js0.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	q_init=curve_js[0]

	opt=lambda_opt(curve,curve_normal)

	lowerer_limit=[-np.pi]
	upper_limit=[np.pi]
	bnds=tuple(zip(lowerer_limit,upper_limit))
	res = differential_evolution(opt.single_arm_theta0_opt, bnds, args=None,workers=1,
									x0 = [0],
									strategy='best1bin', maxiter=2000,
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

	dlam_out=calc_lamdot(q_out,opt.lam[:len(q_out)],opt.joint_vel_limit,1)


	plt.plot(opt.lam[:len(q_out)-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("velocity-constraint_js.png")
	plt.show()
	

if __name__ == "__main__":
	main()