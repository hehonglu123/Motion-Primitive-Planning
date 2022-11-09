import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import os
import yaml

sys.path.append('../../../constraint_solver')
sys.path.append('../fanuc_toolbox')
from constraint_solver import *
from fanuc_utils import *
from error_check import *
from utils import *

def main():

    # data_type='curve_1'
    data_type='curve_2_scale'

    data_dir='../../../data/'+data_type+'/'
    output_dir='../data/'+data_type+'/dual_arm_de/'
    
    relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

    # ###### get tcp
    #read in initial curve pose
    # with open('../data/'+data_type+'/blade_pose.yaml') as file:
    #     blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
    # scale=0.85
    # H_200id=np.loadtxt(output_dir+'tcp.csv',delimiter=',')
    # bT = Transform(H_200id[:3,:3],H_200id[:3,3]*scale)
    # bT=Transform(R=np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),p=[0,0,0]).inv()*bT
    # print(bT)
    # print(R2wpr(bT.R))
    # bT.p[2]=-420
    # bT=Transform(R=Rz(np.radians(180)),p=[0,0,0]).inv()*bT
    # print(bT)
    # exit()
    # robot=m900ia(R_tool=np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),p_tool=np.array([0,0,0])*1000.,d=0)
    # # T_tool=robot.fwd(np.deg2rad([0,49.8,-17.2,0,65.4,0]))
    # # T_tool=robot.fwd(np.deg2rad([0,48.3,-7,0,55.8,0]))
    # # T_tool=robot.fwd(np.deg2rad([0,0,0,0,0,0]))
    # # T_tool=robot.fwd(np.deg2rad([0,40.5,-28.8,0,69.3,0]))
    # T_tool=Transform(np.matmul(Ry(np.radians(-90)),Rx(np.radians(180))),[1950,0,650])
    # bT=T_tool.inv()*bT
    # print(bT)
    # print(R2wpr(bT.R))
    # bT=Transform(np.matmul(Ry(np.radians(90)),Rz(np.radians(180))),np.array([0,0,0])*1000.)*bT
    # # exit()
    # bT_T=np.vstack((np.vstack((bT.R.T,bT.p)).T,[0,0,0,1]))
    # # print(bT_T)
    # with open(data_dir+'tcp.yaml','w') as file:
    #     yaml.dump({'H':bT_T.tolist()},file)
    # exit()
    # ##################################################

    # curve 1
    # v_cmd=500
    # curve 2 scale
    v_cmd=800

    H_200id=np.loadtxt(output_dir+'H_lrmate200id.csv',delimiter=',')

    base2_R=H_200id[:3,:3]
    base2_p=H_200id[:-1,-1]

    base2_k,base2_theta=R2rot(base2_R)

    toolbox_path = '../../../toolbox/'
    robot1=robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    robot2=robot_obj('FANUC_lrmate_200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path=output_dir+'tcp.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])

    opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=50000,v_cmd=v_cmd)

    ## fwd check
    # print(robot2.fwd(np.radians([0,0,0,0,30,-90])))
    # print(R2wpr(robot2.fwd(np.radians([0,0,0,0,30,-90])).R))

    ## find valid x y q_init2
    q_init2_init=np.radians([0,15,-15,0,15,-90])
    # print(robot2.fwd(q_init2_init))
    # print(q_init2_init)
    # print(base2_p)
    # print(base2_theta)

    input_x = np.append(q_init2_init,base2_p[:2])
    input_x = np.append(input_x,base2_theta)
    input_x = np.append(input_x,0)
    print(input_x)
    print(opt.dual_arm_opt_w_pose_3dof(input_x))
    exit()
    ###########################################diff evo opt############################################
    ##x:q_init2,base2_x,base2_y,base2_theta,theta_0
    q_init2_init=np.radians([0,0,0,0,-30,90])
    lower_limit=np.hstack((robot2.lower_limit,[0,0],[-np.pi],[-np.pi]))
    upper_limit=np.hstack((robot2.upper_limit,[3000,3000],[np.pi],[np.pi]))
    bnds=tuple(zip(lower_limit,upper_limit))
    res = differential_evolution(opt.dual_arm_opt_w_pose_3dof, bnds, args=None,workers=-1,
                                    x0 = np.hstack((q_init2_init,base2_p[0],base2_p[1],base2_theta,[0])),
                                    strategy='best1bin', maxiter=700,
                                    popsize=15, tol=1e-10,
                                    mutation=(0.5, 1), recombination=0.7,
                                    seed=None, callback=None, disp=True,
                                    polish=True, init='latinhypercube',
                                    atol=0)
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

    robot2.base_H=H_from_RT(base2_R,base2_p)
    pose2_world_now=robot2.fwd(q_init2,world=True)


    R_temp=direction2R(pose2_world_now.R@opt.curve_normal[0],-opt.curve[1]+opt.curve[0])
    R=np.dot(R_temp,Rz(res.x[-1]))

    q_init1=robot1.inv(pose2_world_now.p,R)[0]

    opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=50000)
    q_out1,q_out2=opt.dual_arm_stepwise_optimize(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,w1=0.02,w2=0.01)

    ####output to trajectory csv
    df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
    df.to_csv(output_dir+'arm1.csv',header=False,index=False)
    df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
    df.to_csv(output_dir+'arm2.csv',header=False,index=False)

    ###output to pose yaml
    H=np.eye(4)
    H[:-1,-1]=base2_p
    H[:3,:3]=base2_R

    np.savetxt(output_dir+'base.csv', H, delimiter=',')

    ###dual lambda_dot calc
    speed,speed1,speed2=traj_speed_est_dual(robot1,robot2,q_out1[::100],q_out2[::100],opt.lam[::100],v_cmd)

    print('speed min: ', min(speed))

    plt.plot(opt.lam[::100],speed,label="lambda_dot_max")
    plt.xlabel("lambda")
    plt.ylabel("lambda_dot")
    plt.title("DUALARM max lambda_dot vs lambda (path index)")
    plt.ylim([0,1.2*v_cmd])
    plt.savefig(output_dir+"results.png")
    # plt.show()


if __name__ == "__main__":
    main()