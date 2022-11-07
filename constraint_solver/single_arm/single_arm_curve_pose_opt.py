import sys, yaml
sys.path.append('../')
from constraint_solver import *
sys.path.append('../../')
from tes_env import *

def main():
	dataset='curve_1'
	data_dir='../../data/'+dataset+'/'
	###read actual curve
	curve_dense = read_csv(data_dir+"Curve_dense.csv",header=None).values


	# robot=robot_obj('ABB_6640_180_255','../../config/ABB_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../toolbox/robot_info/6640acc_new.pickle')
	robot = robot_obj('FANUC_m10ia','../../config/FANUC_m10ia_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../config/FANUC_m10ia_acc_new.pickle')

	v_cmd=500
	opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,urdf_path='../../config/urdf/',curve_name=dataset,steps=500,v_cmd=v_cmd)

	#read in initial curve pose
	# with open(data_dir+'baseline/curve_pose.yaml') as file:
	# 	curve_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

	curve_pose_fanuc=\
	np.array([[-0.008281240596627095,-0.9895715673738508,0.14380380418973931,700.2585970910012],\
	[0.9999617720243631,-0.007791538934054126,0.00396817476110772,-506.82332718104914],\
	[-0.0028063399787532726,0.14383116827134246,0.9895982616646132,566.410804375142],\
	[0,0,0,1]])
	curve_pose=curve_pose_fanuc

	k,theta=R2rot(curve_pose[:3,:3])

	###path constraints, position constraint and curve normal constraint
	lowerer_limit=np.array([-2*np.pi,-2*np.pi,-2*np.pi,0,-3000,0,-np.pi])
	upper_limit=np.array([2*np.pi,2*np.pi,2*np.pi,3000,3000,3000,np.pi])
	bnds=tuple(zip(lowerer_limit,upper_limit))


	res = differential_evolution(opt.curve_pose_opt2, bnds, args=None,workers=11,
									x0 = np.hstack((k*theta,curve_pose[:-1,-1],[0])),
									strategy='best1bin', maxiter=10,
									popsize=15, tol=1e-10,
									mutation=(0.5, 1), recombination=0.7,
									seed=None, callback=None, disp=True,
									polish=True, init='latinhypercube',
									atol=0.)
	
	# class Object(object):
	# 	pass
	# res=Object()
	# setattr(res, "x", np.array([ 5.22786277e+00,  4.30371672e+00,  3.25477272e+00,  1.25793114e+03, 7.65140344e+02,  6.87793008e+02, -2.82344434e-01]))


	print(res)
	theta0=np.linalg.norm(res.x[:3])
	k=res.x[:3]/theta0
	shift=res.x[3:-1]
	theta1=res.x[-1]

	R_curve=rot(k,theta0)
	curve_pose=np.vstack((np.hstack((R_curve,np.array([shift]).T)),np.array([0,0,0,1])))

	with open(r'trajectory/curve_pose_opt/curve_pose.yaml', 'w') as file:
		documents = yaml.dump({'H':curve_pose.tolist()}, file)
	###get initial q
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

	R_temp=direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
	R=np.dot(R_temp,Rz(theta1))
	q_init=robot.inv(curve_new[0],R)[0]

	#########################################restore only given points, saves time##########################################################
	q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
	# q_out=opt.followx(curve_new,curve_normal_new)

	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/curve_pose_opt/arm1.csv',header=False,index=False)
	df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
	df.to_csv('trajectory/curve_pose_opt/curve_pose_opt_cs.csv',header=False,index=False)
	#########################################restore only given points, END##########################################################

	# dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)
	speed=traj_speed_est(opt.robot1,q_out,opt.lam,opt.v_cmd)

	# plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
	plt.plot(opt.lam,speed,label="speed est")
	plt.xlabel("lambda")
	plt.ylabel("lambda_dot")
	# plt.ylim([1000,4000])
	plt.title("max lambda_dot vs lambda (path index)")
	plt.savefig("trajectory/curve_pose_opt/results.png")

	###optional, solve for dense curve
	#########################################restore all 50,000 points, takes time##########################################################
	opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,steps=50000,v_cmd=v_cmd)
	curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
	curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

	q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
	####output to trajectory csv
	df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
	df.to_csv('trajectory/curve_pose_opt/Curve_js.csv',header=False,index=False)
	df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
	df.to_csv('trajectory/curve_pose_opt/Curve_in_base_frame.csv',header=False,index=False)
	#########################################restore all 50,000 points, END##########################################################
	
	
	

	

if __name__ == "__main__":
	main()