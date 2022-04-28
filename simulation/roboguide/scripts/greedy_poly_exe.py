from math import radians
import numpy as np
from pandas import read_csv

from general_robotics_toolbox import *
from general_robotics_toolbox.general_robotics_toolbox_invkin import *
import sys

# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *
sys.path.append('../../../constraint_solver')
from qp_resolve import *
sys.path.append('../../../greedy_fitting')
from greedy_poly import *

def main():
    
    # define m900ia
    robot = m900ia(d=50)
    utool_num = 2

    # the original curve in Cartesian space
    col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
    data = read_csv("../../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
    curve_x=data['X'].tolist()
    curve_y=data['Y'].tolist()
    curve_z=data['Z'].tolist()
    curve_direction_x=data['direction_x'].tolist()
    curve_direction_y=data['direction_y'].tolist()
    curve_direction_z=data['direction_z'].tolist()
    curve=np.vstack((curve_x, curve_y, curve_z)).T
    curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

    ###read in poly
    col_names=['poly_x', 'poly_y', 'poly_z','poly_direction_x', 'poly_direction_y', 'poly_direction_z'] 
    data = read_csv("../../../data/from_ge/Curve_in_base_frame_poly.csv", names=col_names)
    poly_x=data['poly_x'].tolist()
    poly_y=data['poly_y'].tolist()
    poly_z=data['poly_z'].tolist()
    curve_poly_coeff=np.vstack((poly_x, poly_y, poly_z))

    col_names=['poly_q1', 'poly_q2', 'poly_q3','poly_q4', 'poly_q5', 'poly_q6'] 
    data = read_csv("../../../data/from_ge/Curve_js_poly.csv", names=col_names)
    poly_q1=data['poly_q1'].tolist()
    poly_q2=data['poly_q2'].tolist()
    poly_q3=data['poly_q3'].tolist()
    poly_q4=data['poly_q4'].tolist()
    poly_q5=data['poly_q5'].tolist()
    poly_q6=data['poly_q6'].tolist()
    curve_js_poly_coeff=np.vstack((poly_q1, poly_q2, poly_q3,poly_q4,poly_q5,poly_q6))

    # save directory
    save_dir='../data/greedy_poly/'

    greedy_fit_obj=greedy_fit(robot,curve_poly_coeff,curve_js_poly_coeff, num_points=500, orientation_weight=1)
    breakpoints,primitives_choices,points=greedy_fit_obj.fit_under_error(0.5)
    # print(breakpoints)
    # print(primitives_choices)
    # print(points)
    # print(greedy_fit_obj.curve_fit_js.shape)
    curve_fit_js=greedy_fit_obj.curve_fit_js
    curve_fit=greedy_fit_obj.curve_fit
    curve_fit_R=greedy_fit_obj.curve_fit_R

    # print(np.rad2deg(curve_fit_js[np.array(breakpoints[1:]).astype(int)-1]))

    curve_fit_js_unwrap=[curve_fit_js[0]]
    for qi in range(1,len(curve_fit_js)):
        T=fwdkin(robot.robot_def,curve_fit_js[qi])
        this_q=robot6_sphericalwrist_invkin(robot.robot_def,T,curve_fit_js_unwrap[-1])
        curve_fit_js_unwrap.append(this_q[0])

    # for bp in range(breakpoints[-3],breakpoints[-2]):
    #     print(np.rad2deg(curve_fit_js_unwrap[bp]))

    # print(points)
    # print(curve_fit[np.array(breakpoints[1:]).astype(int)-1])
    # print(curve_fit[breakpoints[1]-1])
    # print(points[0][0])
    # for bpi in range(0,breakpoints[2]-1):
    #     if np.all(curve_fit[bpi]==points[1][0]):
    #         print(bpi)
    #         print(curve_fit[bpi])
    #         break

    # define speed and zone (CNT)
    speed = 2000 # max is 2000 mmsec
    j_speed = 100 # max is 100 %
    # zone = 100

    # fanuc client
    client = FANUCClient()

    # all_zones=[25,50,75,100]
    all_zones=[25,50]
    # all_zones=[100]

    for zone in all_zones:

        # tp program
        # move to start
        tp_pre = TPMotionProgram()
        j0 = jointtarget(1,1,utool_num,np.degrees(curve_fit_js_unwrap[0]),[0]*6)
        tp_pre.moveJ(j0,50,'%',-1)
        client.execute_motion_program(tp_pre)

        # start moving along the curve: moveL
        tp = TPMotionProgram()
        for mo_id in range(0,len(breakpoints)-1):
            bp=breakpoints[mo_id+1]
            this_motion=primitives_choices[mo_id]
            this_zone=zone
            if mo_id==len(breakpoints)-2:
                this_zone=-1

            if this_motion == 'movej_fit':
                j0 = jointtarget(1,1,utool_num,np.degrees(curve_fit_js_unwrap[bp-1]),[0]*6)
                tp.moveJ(j0,j_speed,'%',this_zone)
            elif this_motion == 'movel_fit':
                robt = joint2robtarget(curve_fit_js_unwrap[bp-1],robot,1,1,2)
                tp.moveL(robt,speed,'mmsec',this_zone)
            elif this_motion == 'movec_fit':
                robt2 = joint2robtarget(curve_fit_js_unwrap[bp-1],robot,1,1,2) # goal
                robt1 = joint2robtarget(curve_fit_js_unwrap[int((bp-breakpoints[mo_id])/2+breakpoints[mo_id])],robot,1,1,2) # mid point
                # robt2 = jointtarget(1,1,utool_num,np.degrees(curve_fit_js_unwrap[bp-1]),[0]*6) # goal
                # robt1 = jointtarget(1,1,utool_num,np.degrees(curve_fit_js_unwrap[int((bp-breakpoints[mo_id])/2+breakpoints[mo_id])]),[0]*6) # mid point
                tp.moveC(robt1,robt2,speed,'mmsec',this_zone)
        # execute 
        res = client.execute_motion_program(tp)
        # Write log csv to file
        with open(save_dir+"greedy_poly_"+str(zone)+".csv","wb") as f:
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

