import sys
sys.path.append('../')
sys.path.append('../../toolbox/')
from constraint_solver import *

def main():

	data_dir='../../data/curve_2/'
	solution_dir=data_dir+'dual_arm/diffevo_pose6_3/'
	# data_dir='../../data/curve_1/'
	# solution_dir=data_dir+'dual_arm/diffevo_pose3_2/'
	relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

	v_cmd=3666
	# v_cmd=1333

	H_1200=np.loadtxt(data_dir+'dual_arm/abb1200.csv',delimiter=',')

	base2_R=H_1200[:3,:3]
	base2_p=H_1200[:-1,-1]

	base2_k,base2_theta=R2rot(base2_R)

	robot1=robot_obj('ABB_6640_180_255','../../config/abb_6640_180_255_robot_default_config.yml',tool_file_path='../../config/paintgun.csv',d=50,acc_dict_path='../../config/acceleration/6640acc_new.pickle')
	robot2=robot_obj('ABB_1200_5_90','../../config/abb_1200_5_90_robot_default_config.yml',tool_file_path=data_dir+'dual_arm/tcp.csv',acc_dict_path='../../config/acceleration/1200acc_new.pickle')

	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=500,v_cmd=v_cmd)

	q_init1=np.loadtxt(solution_dir+'arm1.csv',delimiter=',')[0]
	q_init2=np.loadtxt(solution_dir+'arm2.csv',delimiter=',')[0]
	base_H=np.loadtxt(solution_dir+'base.csv',delimiter=',')
	base2_p=base_H[:-1,-1]
	base2_R=base_H[:3,:3]

	opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=500)
	lam=calc_lam_cs(relative_path[:,:3])

	q_out1,q_out2,_,_=opt.dual_arm_stepwise_optimize(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,w1=0.01,w2=0.01,using_spherical=True)
	# q_out1_new,q_out2_new,_,_=opt.dual_arm_stepwise_optimize2(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,lamdot_des=777,w1=0.01,w2=0.01,using_spherical=False)
	q_out1_new,q_out2_new,_,_=opt.dual_arm_stepwise_optimize3(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,lamdot_des=1000,w1=0.01,w2=0.01)


	###dual lambda_dot calc
	lamdot_boundary=lambdadot_qlambda_dual(robot1,robot2,q_out1,q_out2,opt.lam)
	lamdot_boundary_new=lambdadot_qlambda_dual(robot1,robot2,q_out1_new,q_out2_new,opt.lam)

	print(opt.v_cmd)
	speed, _, _=traj_speed_est_dual(opt.robot1,opt.robot2,q_out1,q_out2,opt.lam,v_cmd)
	speed_new, _, _=traj_speed_est_dual(opt.robot1,opt.robot2,q_out1_new,q_out2_new,opt.lam,v_cmd)

	plt.plot(opt.lam,lamdot_boundary,label=r'$\dot{\lambda}$ boundary old')
	plt.plot(opt.lam,lamdot_boundary_new,label=r'$\dot{\lambda}$ boundary new')

	plt.plot(opt.lam,speed,label=r'speed estimation old')
	plt.plot(opt.lam,speed_new,label=r'speed estimation new')

	plt.legend()
	plt.xlabel(r'$\lambda$ (mm)')
	plt.ylabel(r'$\dot{\lambda}$ (mm/s)')
	plt.title(r'$\dot{\lambda}$ Boundary Profile' )
	plt.show()


if __name__ == "__main__":
	main()