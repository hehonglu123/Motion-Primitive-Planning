import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import os
import yaml

sys.path.append('../../../constraint_solver')
from constraint_solver import *


def main():
    # data_type='wood'
    data_type='blade_scale'

    if data_type=='blade_scale':
        curve_data_dir='../../../data/from_NX_scale/'
        data_dir='../data/curve_blade_scale/'
    elif data_type=='wood':
        curve_data_dir='../../../data/wood/'
        data_dir='../data/curve_wood/'

    ###read actual curve
    curve_dense = read_csv(curve_data_dir+"Curve_dense.csv",header=None).values

    # robot=m10ia(d=50, acc_dict_path=os.getcwd()+'/../../../toolbox/robot_info/m10ia_acc.pickle')
    toolbox_path = '../../../toolbox/'
    robot = robot_obj(toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    v_cmd=350
    opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,steps=500,v_cmd=v_cmd)

    #read in initial curve pose
    with open(data_dir+'blade_pose.yaml') as file:
        curve_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

    k,theta=R2rot(curve_pose[:3,:3])

    ###path constraints, position constraint and curve normal constraint
    lowerer_limit=np.array([-2*np.pi,-2*np.pi,-2*np.pi,0,-3000,0,-np.pi])
    upper_limit=np.array([2*np.pi,2*np.pi,2*np.pi,3000,3000,3000,np.pi])
    bnds=tuple(zip(lowerer_limit,upper_limit))

    ###diff evolution
    # res = differential_evolution(opt.curve_pose_opt2, bnds, args=None,workers=1,
	# 								x0 = np.hstack((k*theta,curve_pose[:-1,-1],[0])),
	# 								strategy='best1bin', maxiter=1,
    #                                 # strategy='best1bin', maxiter=1,
	# 								popsize=15, tol=1e-10,
	# 								mutation=(0.5, 1), recombination=0.7,
	# 								seed=None, callback=None, disp=False,
	# 								polish=True, init='latinhypercube',
	# 								atol=0.)
    # print(res)
    # theta0=np.linalg.norm(res.x[:3])
    # k=res.x[:3]/theta0
    # shift=res.x[3:-1]
    # theta1=res.x[-1]
    # R_curve=rot(k,theta0)
    # curve_pose=np.vstack((np.hstack((R_curve,np.array([shift]).T)),np.array([0,0,0,1])))
    # with open(data_dir+'curve_pose.yaml', 'w') as file:
    #     documents = yaml.dump({'H':curve_pose.tolist()}, file)
    
    R_curve=curve_pose[:3,:3]
    shift=curve_pose[:3,3]
    theta1=0

    ###get initial q
    curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
    curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

    R_temp=direction2R(curve_normal_new[0],-curve_new[1]+curve_new[0])
    R=np.dot(R_temp,Rz(theta1))
    print(robot.inv(curve_new[0],R))
    q_init=robot.inv(curve_new[0],R)[0]

    #########################################restore only given points, saves time##########################################################
    q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
    # q_out=opt.followx(curve_new,curve_normal_new)

    ####output to trajectory csv
    df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
    df.to_csv(data_dir+'single_arm_de/arm1.csv',header=False,index=False)
    df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
    df.to_csv(data_dir+'single_arm_de/curve_pose_opt_cs.csv',header=False,index=False)
    #########################################restore only given points, END##########################################################

    # dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)
    speed=traj_speed_est(opt.robot1,q_out,opt.lam,opt.v_cmd)

    # plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
    plt.plot(opt.lam,speed,label="speed est")
    plt.xlabel("lambda")
    plt.ylabel("lambda_dot")
    # plt.ylim([1000,4000])
    plt.title("max lambda_dot vs lambda (path index)")
    plt.savefig(data_dir+"single_arm_de/results.png")

    ###optional, solve for dense curve
    #########################################restore all 50,000 points, takes time##########################################################
    opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,steps=50000,v_cmd=v_cmd)
    curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
    curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

    q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
    ####output to trajectory csv
    df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
    df.to_csv(data_dir+'single_arm_de/Curve_js.csv',header=False,index=False)
    df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
    df.to_csv(data_dir+'single_arm_de/Curve_in_base_frame.csv',header=False,index=False)
    #########################################restore all 50,000 points, END##########################################################


if __name__ == "__main__":
    main()