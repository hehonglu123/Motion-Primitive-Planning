import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import os
import yaml

# sys.path.append('..constraint_solver')
# sys.path.append('../fanuc_toolbox')
from constraint_solver import *
# from fanuc_utils import *
from error_check import *
from utils import *

class Geeks():
    x: np.array

def main():

    data_type='curve_1'
    # data_type='curve_2_scale'

    print(data_type)

    # data_dir='../../../data/'+data_type+'/'
    data_dir=data_type+'/'
    output_dir=data_type+'/dual_arm_de/'
    
    relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

    # curve 1
    # v_cmd=500
    # curve 2 scale
    v_cmd=800

    H_200id=np.loadtxt(output_dir+'H_lrmate200id.csv',delimiter=',')

    base2_R=H_200id[:3,:3]
    base2_p=H_200id[:-1,-1]

    print(base2_R,base2_p)

    base2_k,base2_theta=R2rot(base2_R)

    # toolbox_path = ''
    # robot1=robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    # robot2=robot_obj('FANUC_lrmate_200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path=output_dir+'tcp.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    # robot1=robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc_compensate.pickle')
    # robot2=robot_obj('FANUC_lrmate_200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path=output_dir+'tcp.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc_compensate.pickle')

    robot2_tcp=np.loadtxt(output_dir+'tcp.csv',delimiter=',')

    robot1=m10ia(d=50,acc_dict_path='robot_info/m10ia_acc.pickle')
    robot2=lrmate200id(R_tool=Ry(np.pi/2)@robot2_tcp[:3,:3],p_tool=Ry(np.pi/2)@robot2_tcp[:3,-1],acc_dict_path='robot_info/lrmate200id_acc.pickle')

    opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=500,v_cmd=v_cmd)

    ## fwd check
    # print(robot1.fwd(np.radians([0,0,10,10,30,-90])))
    # print(robot2.fwd(np.radians([0,0,10,10,30,-90])))
    # print(R2wpr(robot2.fwd(np.radians([0,0,0,0,30,-90])).R))

    ## find valid x y q_init2
    # with open(data_type+'/blade_pose.yaml') as file:
    #     blade_pose = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
    # blade_pose_base2=Transform(base2_R,base2_p).inv()*Transform(blade_pose[:3,:3],blade_pose[:-1,-1])
    # print(blade_pose_base2)
    # print(np.degrees(robot2.inv(p=blade_pose_base2.p,R=blade_pose_base2.R)))
    # exit()
    # q_init2_init=np.radians([0,15,-15,0,15,-90])
    # q_init2_init=robot2.inv(p=blade_pose_base2.p,R=blade_pose_base2.R)[0]
    # print(robot2.fwd(q_init2_init))
    # print(q_init2_init)
    # print(base2_p)
    # print(base2_theta)

    # print(Transform(base2_R,base2_p)*robot2.fwd(q_init2_init))
    # print(blade_pose)
    # exit()

    q_init2_init=np.radians([0,10,10,0,-30,90])
    input_x = np.append(q_init2_init,base2_p[:2])
    input_x = np.append(input_x,base2_theta)
    input_x = np.append(input_x,0)
    # print(input_x)
    print("Sanity Check")
    print(opt.dual_arm_opt_w_pose_3dof(input_x))
    print("Sanity Check Done")
    # exit()
    ###########################################diff evo opt############################################
    ##x:q_init2,base2_x,base2_y,base2_theta,theta_0
    lower_limit=np.hstack((robot2.lower_limit,[-300,-2000],[-np.pi],[-np.pi]))
    upper_limit=np.hstack((robot2.upper_limit,[2000,2000],[np.pi],[np.pi]))
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
    # res.x=np.array([1.93308869e-02, -9.80895904e-01, -7.26287729e-01,  2.54798516e-01,\
    #    -8.50067936e-01,  4.85071875e+00,  9.53418192e+02,  5.83962951e+02,\
    #     8.85032818e-01,  2.36613695e+00])
    # print(opt.dual_arm_opt_w_pose_3dof(res.x))
    # exit()
    # res=Geeks()
    # res.x=np.array([-1.09604735e+00,  1.89831661e+00,  2.33015474e+00,  6.34691327e-01,\
    #    -1.57684494e+00,  4.87632099e+00,  7.68141579e+02,  3.53649048e+02,\
    #     2.97814076e+00, -2.58943832e+00])
    # print(opt.dual_arm_opt_w_pose_3dof(res.x))
    # exit()

    q_init2=res.x[:6]
    base2_p=np.array([res.x[6],res.x[7],0])		###fixed z height
    base2_theta=res.x[8]
    base2_R=Rz(base2_theta)

    robot2.base_H=H_from_RT(base2_R,base2_p)
    pose2_world_now=robot2.fwd(q_init2,world=True)

    R_temp=direction2R(pose2_world_now.R@opt.curve_normal[0],pose2_world_now.R@(-opt.curve[1]+opt.curve[0]))
    R=np.dot(R_temp,Rz(res.x[-1]))

    # q_init1=robot1.inv(pose2_world_now.p,R)[0]
    q_init1=robot1.inv(np.matmul(pose2_world_now.R,opt.curve[0])+pose2_world_now.p,R)[0]

    opt=lambda_opt(relative_path[:,:3],relative_path[:,3:],robot1=robot1,robot2=robot2,steps=50000)
    q_out1,q_out2,j_out1,j_out2=opt.dual_arm_stepwise_optimize(q_init1,q_init2,base2_R=base2_R,base2_p=base2_p,w1=0.01,w2=0.02)

    jac_check_count=500
    jminall1=[]
    jminall2=[]
    for J in j_out1[::jac_check_count]:
        _,sv,_=np.linalg.svd(J)
        jminall1.append(sv)
    for J in j_out2[::jac_check_count]:
        _,sv,_=np.linalg.svd(J)
        jminall2.append(sv)
    print("J1 min svd:",np.min(jminall1))
    print("J2 min svd:",np.min(jminall2))

    ####output to trajectory csv
    df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
    df.to_csv(output_dir+'arm1.csv',header=False,index=False)
    df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
    df.to_csv(output_dir+'arm2.csv',header=False,index=False)
    # df=DataFrame({'q0':q_out1[:,0],'q1':q_out1[:,1],'q2':q_out1[:,2],'q3':q_out1[:,3],'q4':q_out1[:,4],'q5':q_out1[:,5]})
    # df.to_csv(output_dir+'arm1.csv',header=False,index=False)
    # df=DataFrame({'q0':q_out2[:,0],'q1':q_out2[:,1],'q2':q_out2[:,2],'q3':q_out2[:,3],'q4':q_out2[:,4],'q5':q_out2[:,5]})
    # df.to_csv(output_dir+'arm2.csv',header=False,index=False)


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