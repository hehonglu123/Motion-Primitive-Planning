import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import os
import yaml

sys.path.append('../../../constraint_solver')
from constraint_solver import *
sys.path.append('../../../toolbox')
from tes_env import *

class Geeks():
    x: np.array

def main():
    # data_type='curve_1'
    # data_type='curve_2_scale'

    data_type='curve_1_half'
    # data_type='curve_2_scale_half'

    ## using our paintgun
    # tooltype='paintgun'
    ## using GE laser
    tooltype='laser_ge'

    # data and curve directory
    curve_data_dir='../../../data/'+data_type+'/'
    data_dir='../data/'+data_type+'/'

    ## robot case
    robot_case='single_arm_de_'+tooltype
    output_dir=data_dir+robot_case+'/'

    ###read actual curve
    curve_dense = read_csv(curve_data_dir+"Curve_dense.csv",header=None).values

    toolbox_path = '../../../toolbox/'
    robot_name='FANUC_m10ia'
    robot = robot_obj(robot_name,toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/'+tooltype+'.csv',d=0,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle')

    v_cmd=350
    opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,steps=500,v_cmd=v_cmd)

    tes_env=Tess_Env('../../../config/urdf/')
    tes_env.update_pose(robot_name,np.eye(4))
    print('tes done')
    print(tes_env.check_collision_single(robot_name,data_type,np.array([[0,0,0,0,0,0],[1,0,0,0,0,0]]).astype(float)))
    # time.sleep(300)

    #read in initial curve pose
    # with open(data_dir+'blade_pose.yaml') as file:
    #     curve_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
    curve_pose=np.loadtxt(data_dir+'curve_pose.csv',delimiter=',')

    k,theta=R2rot(curve_pose[:3,:3])

    ###path constraints, position constraint and curve normal constraint
    lowerer_limit=np.array([-2*np.pi,-2*np.pi,-2*np.pi,0,-3000,0,-np.pi])
    upper_limit=np.array([2*np.pi,2*np.pi,2*np.pi,3000,3000,3000,np.pi])
    bnds=tuple(zip(lowerer_limit,upper_limit))

    x_init = np.hstack((k*theta,curve_pose[:-1,-1],[0]))
    print(x_init)

    use_tes=False
    if use_tes:
        opt.tes_env=tes_env

    print("Sanity Check")
    print(opt.curve_pose_opt2(x_init))
    print("Sanity Check Done")

    def print_cb(xk,convergence):
        print(xk)

    ###diff evolution
    # res = differential_evolution(opt.curve_pose_opt2, bnds, args=None,workers=1,
	# 								x0 = x_init,
	# 								strategy='best1bin', maxiter=1,
	# 								popsize=15, tol=1e-10,
	# 								mutation=(0.5, 1), recombination=0.7,
	# 								seed=None, callback=print_cb, disp=False,
	# 								polish=True, init='latinhypercube',
	# 								atol=0.)

    res=Geeks()
    res.x=np.array([2.90640650e+00, -4.20905594e+00,  7.27557922e-01,  4.75611551e+02,
        4.67387948e+02,  8.16317533e+02, -3.00079346e+00])

    print(res)
    theta0=np.linalg.norm(res.x[:3])	###pose rotation angle
    k=res.x[:3]/theta0					###pose rotation axis
    shift=res.x[3:-1]					###pose translation
    theta1=res.x[-1]	
    R_curve=rot(k,theta0)
    curve_pose=np.vstack((np.hstack((R_curve,np.array([shift]).T)),np.array([0,0,0,1])))
    with open(data_dir+'curve_pose.yaml', 'w') as file:
        documents = yaml.dump({'H':curve_pose.tolist()}, file)

    ###get initial q
    curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
    curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

    curve_origin_0_new = np.dot(R_curve,opt.curve_original[0])+shift
    curve_origin_1_new = np.dot(R_curve,opt.curve_original[1])+shift
    # R_temp=direction2R(curve_normal_base[0],-curve_base[1]+curve_base[0])
    R_temp=direction2R_Y(curve_normal_new[0],curve_origin_1_new-curve_origin_0_new)
    R=np.dot(R_temp,Rz(theta1))
    q_init=robot.inv(curve_new[0],R)[0]

    #########################################restore only given points, saves time##########################################################
    q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)
    # q_out=opt.followx(curve_base,curve_normal_new)

    print("Collision? (scarse)")
    print(tes_env.check_collision_single(robot_name,data_type,q_out))
    print("===================")

    ####output to trajectory csv
    df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
    df.to_csv(output_dir+'arm1.csv',header=False,index=False)
    df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
    df.to_csv(output_dir+'curve_pose_opt_cs.csv',header=False,index=False)
    #########################################restore only given points, END##########################################################

    # dlam_out=calc_lamdot(q_out,opt.lam,opt.robot1,1)
    speed=traj_speed_est(opt.robot1,q_out,opt.lam,opt.v_cmd)
    print(min(speed))

    # plt.plot(opt.lam,dlam_out,label="lambda_dot_max")
    plt.plot(opt.lam,speed,label="speed est")
    plt.xlabel("lambda")
    plt.ylabel("lambda_dot")
    # plt.ylim([1000,4000])
    plt.title("max lambda_dot vs lambda (path index)")
    plt.savefig(output_dir+"results.png")

    ###optional, solve for dense curve
    #########################################restore all 50,000 points, takes time##########################################################
    opt=lambda_opt(curve_dense[:,:3],curve_dense[:,3:],robot1=robot,steps=len(opt.curve_original),v_cmd=v_cmd)
    curve_new=np.dot(R_curve,opt.curve.T).T+np.tile(shift,(len(opt.curve),1))
    curve_normal_new=np.dot(R_curve,opt.curve_normal.T).T

    q_out=opt.single_arm_stepwise_optimize(q_init,curve_new,curve_normal_new)

    print("Collision? (dense)")
    print(tes_env.check_collision_single(robot_name,data_type,q_out))
    print("===================")

    ####output to trajectory csv
    df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
    df.to_csv(output_dir+'Curve_js.csv',header=False,index=False)
    df=DataFrame({'x':curve_new[:,0],'y':curve_new[:,1],'z':curve_new[:,2],'nx':curve_normal_new[:,0],'ny':curve_normal_new[:,1],'nz':curve_normal_new[:,2]})
    df.to_csv(output_dir+'Curve_in_base_frame.csv',header=False,index=False)
    #########################################restore all 50,000 points, END##########################################################


if __name__ == "__main__":
    main()