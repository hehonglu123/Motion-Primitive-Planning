import sys, yaml
sys.path.append('../')
from constraint_solver import *

def main():
	###read actual curve
	col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
	data = data = read_csv("../../data/from_ge/relative_path.csv", names=col_names)
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


	###path constraints, position constraint and curve normal constraint

	lowerer_limit=np.array([-999,-999,-999,-3000,-3000,-3000,-np.pi])
	upper_limit=-lowerer_limit
	bnds=tuple(zip(lowerer_limit,upper_limit))


	# res = minimize(opt.curve_pose_opt, [0.57735027, 0.57735027, 0.57735027,2.0943951023931957,2700,-800,500,0], method='SLSQP',tol=1e-10,bounds=bnds)

	res = differential_evolution(opt.curve_pose_opt, bnds, args=None,workers=-1,
									x0 = [1.20919957785,1.20919957785,1.20919957785,2700,-800,500,0],
									strategy='best1bin', maxiter=200,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=False,
									polish=True, init='latinhypercube',
									atol=0.)
	


	print(res)
	theta0=np.linalg.norm(res.x[:3])
	k=res.x[:3]/np.linalg.norm(res.x[:3])
	shift=res.x[3:-1]
	theta1=res.x[-1]

	R_curve=rot(k,theta0)
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

	curve_pose=np.vstack((np.hstack((R_curve,np.array([shift/1000.]).T)),np.array([0,0,0,1])))


	R_temp=opt.direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
	R=np.dot(R_temp,Rz(theta1))
	q_init=robot.inv(curve_new[0],R)[0]


	q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)

	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/curve_pose_opt/arm1.csv',header=False,index=False)
	with open(r'trajectory/curve_pose_opt/curve_pose.yaml', 'w') as file:
		documents = yaml.dump({'H':curve_pose.tolist()}, file)

	dlam_out=calc_lamdot(q_out,opt.lam[:len(q_out)],opt.robot1,1)


	plt.plot(opt.lam[:len(q_out)-1],dlam_out,label="lambda_dot_max")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	plt.ylim([1000,4000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/curve_pose_opt/results.png")
	plt.show()
if __name__ == "__main__":
	main()