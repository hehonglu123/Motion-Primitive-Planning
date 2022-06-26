########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *

def main():
    ms = MotionSend()
    # data_dir="fitting_output_new/python_qp_movel/"
    # data_dir="../../../data/from_NX/baseline/0.1/"
    data_dir="debug/"

    robot=abb6640(d=50)

    vmax = speeddata(10000,9999999,9999999,999999)
    v680 = speeddata(680,9999999,9999999,999999)
    speed={"v500":v500}
    zone={"z10":z10}

    for s in speed:
        for z in zone: 
            breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(data_dir+"circle_uncertain_commands.csv")

            modified_bp=arc_from_3point(p_bp[8][0],p_bp[9][-1],p_bp[9][0],N=3)
            p_bp[9][0]=modified_bp[1]


            original_points=np.array([p_bp[8][0],p_bp[9][0],p_bp[9][1]])
            original_arc=arc_from_3point(p_bp[8][0],p_bp[9][-1],p_bp[9][0])
            lam=calc_lam_cs(original_arc)
            print('arc length: ',lam)

            p_bp, q_bp = ms.extend(robot, q_bp, primitives, breakpoints, p_bp)
            

            plt.figure()
            ax = plt.axes(projection='3d')
            x_all=np.array([robot.fwd(q_bp[5][0]).R[:,0],robot.fwd(q_bp[6][0]).R[:,0],robot.fwd(q_bp[6][1]).R[:,0]])
            y_all=np.array([robot.fwd(q_bp[5][0]).R[:,1],robot.fwd(q_bp[6][0]).R[:,1],robot.fwd(q_bp[6][1]).R[:,1]])
            normal_all=np.array([robot.fwd(q_bp[5][0]).R[:,-1],robot.fwd(q_bp[6][0]).R[:,-1],robot.fwd(q_bp[6][1]).R[:,-1]])
            ax.scatter(original_points[:,0],original_points[:,1],original_points[:,2], c='gray',label='original')
            ax.plot3D(original_arc[:,0],original_arc[:,1],original_arc[:,2], c='gray',label='original')
            ax.quiver(original_points[:,0],original_points[:,1],original_points[:,2],normal_all[:,0],normal_all[:,1],normal_all[:,2],length=1,color='blue')
            ax.quiver(original_points[:,0],original_points[:,1],original_points[:,2],x_all[:,0],x_all[:,1],x_all[:,2],length=1,color='red')
            ax.quiver(original_points[:,0],original_points[:,1],original_points[:,2],y_all[:,0],y_all[:,1],y_all[:,2],length=1,color='green')
            plt.legend()
            plt.show()
            logged_data= ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,speed[s],zone[z])

            f = open(data_dir+"curve_exe"+"_"+s+"_"+z+".csv", "w")
            f.write(logged_data)
            f.close()

if __name__ == "__main__":
    main()