from math import degrees, radians
import numpy as np
from pandas import read_csv

from general_robotics_toolbox import *
import sys

# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *
sys.path.append('../../../constraint_solver')
from qp_resolve import *

def main():
    
    # define m900ia
    robot1 = m710ic(d=50)
    robot2 = m900ia(d=50)
    utool_num = 2

    ###read in points
    col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
    data = read_csv("../data/from_ge/relative_path_tool_frame.csv", names=col_names)
    curve_x=data['X'].tolist()
    curve_y=data['Y'].tolist()
    curve_z=data['Z'].tolist()
    curve_direction_x=data['direction_x'].tolist()
    curve_direction_y=data['direction_y'].tolist()
    curve_direction_z=data['direction_z'].tolist()
    relative_path=np.vstack((curve_x, curve_y, curve_z)).T
    relative_path_direction=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

    base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    base2_p=np.array([3000,1000,0])

    #################################################calculate initial joint angles through inv#########################

    q_init1=np.array([1.041163124,	0.163121569,	0.060872101,	0.975429159,	-1.159317104,	-1.184301005])
    q_init2=np.array([0.141740433,	0.631713663,	0.458836251,	-0.321985253,	-1.545055101,	0.139048988])

    # save directory
    save_dir='../data/qp_dual_movel/'

    lam=calc_lam_cs(curve)
    lamdot_des=1000
    q1_all,q2_all,lam_out,curve_relative_out,curve_relative_normal_out,act_speed=dual_arm_stepwise_optimize(robot1,robot2,q_init1,q_init2,np.zeros(3),np.eye(3),base2_p,base2_R,lam,lamdot_des
                            ,relative_path,relative_path_direction)
    error = calc_max_error_w_normal(curve_relative_out[2:],relative_path,curve_relative_normal_out[2:],relative_path_direction)
    print("Error Point, Angle",error[0],degrees(error[1]))
    print("Best Speed",np.min(act_speed))

    # define speed and zone (CNT)
    speed = 2000 # max is 2000
    # zone = 100

    # fanuc client
    client = FANUCClient()

    all_step_slices=[10,30,50,100]
    all_zones=[25,50,75,100]
    # all_step_slices=[100]
    # all_zones=[100]

    for step_slice in all_step_slices:
        step=int(len(q_all)/step_slice)

        for zone in all_zones:

            # tp program
            # move to start
            tp_pre = TPMotionProgram()
            j0 = jointtarget(1,1,utool_num,np.degrees(q_all[0]),[0]*6)
            tp_pre.moveJ(j0,50,'%',-1)
            client.execute_motion_program(tp_pre)

            # start moving along the curve: moveL
            tp = TPMotionProgram()
            for i in range(1,len(q_all)-1,step):
                robt = joint2robtarget(q_all[i],robot,1,1,2)
                tp.moveL(robt,speed,'mmsec',zone)
            robt = joint2robtarget(q_all[-1],robot,1,1,2) # the last piece
            tp.moveL(robt,speed,'mmsec',-1)
            
            # execute 
            res = client.execute_motion_program(tp)
            # Write log csv to file
            with open(save_dir+"movel_"+str(step_slice)+"_"+str(zone)+".csv","wb") as f:
                f.write(res)

    # robot_test = m900ia(rox.rot([0,1,0],math.pi/2),np.array([0,0,0]))
    # test_T = robot_test.fwd(np.deg2rad([10,0,0,0,30,60]))
    # print(test_T)
    # print(R2wpr(test_T.R))

    # R_tool = Ry(np.radians(120))
    # p_tool = np.array([0.45,0,-0.05])*1000.
    # d = 50
    # test_T = rox.Transform(R_tool,p_tool+np.dot(R_tool,np.array([0,0,d])))
    # print(test_T)
    # print(test_T.p[0])
    # print(R2wpr(test_T.R))


if __name__=='__main__':
    main()

