from RobotRaconteur.Client import *
import time, argparse, sys
import numpy as np
from pandas import *

sys.path.append('../../toolbox')
from robots_def import *

def main():
    robot1="abb1200"
    robot2="abb6640"
    robot1_tool=abb1200(d=50)
    robot2_tool=abb6640()
    url={robot1:'rr+tcp://localhost:23333?service=robot',robot2:'rr+tcp://localhost:12222?service=robot'}
    filename={robot1:'qp_arm1.csv',robot2:'qp_arm2.csv'}

    #########################################modify robot2 base HERE!!!!!!##########################################
    base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    base2_p=np.array([3000,1000,0])
    ###read actual curve
    col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv("trajectory/dual_arm/"+filename[robot1], names=col_names)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    curve_js1=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

    col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv("trajectory/dual_arm/"+filename[robot2], names=col_names)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    curve_js2=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

    total_step=32
    idx=np.linspace(0,len(curve_js1)-1,total_step).astype(int)

    p_prev=np.zeros(3)
    ###find path length to sync motion in trajectory mode
    lam=[0]
    for i in range(len(curve_js1)-1):
        p_new=robot1_tool.fwd(curve_js1[i+1]).p-robot2_tool.fwd(curve_js2[i+1],base2_R,base2_p).p
        lam.append(lam[-1]+np.linalg.norm(p_new-p_prev))
        p_prev=p_new
    ###normalize lam, 
    lam=np.array(lam)/lam[-1]

    robot1_obj = RRN.ConnectService(url[robot1])

    robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", robot1_obj)

    joint_names = [j.joint_identifier.name for j in robot1_obj.robot_info.joint_info]

    halt_mode = robot_const["RobotCommandMode"]["halt"]
    trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
    jog_mode = robot_const["RobotCommandMode"]["jog"]

    JointTrajectoryWaypoint = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectoryWaypoint",robot1_obj)
    JointTrajectory = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectory",robot1_obj)

    ###jog to start pose first
    robot1_obj.command_mode = halt_mode
    time.sleep(0.1)
    robot1_obj.command_mode =jog_mode
    robot1_obj.jog_freespace(curve_js1[0], np.ones(6), True)

    ###switch to traj mode
    robot1_obj.command_mode = halt_mode
    time.sleep(0.1)
    robot1_obj.command_mode = trajectory_mode


    state_w1 = robot1_obj.robot_state.Connect()

    state_w1.WaitInValueValid()

    waypoints = []


    for i in idx:
        wp = JointTrajectoryWaypoint()
        wp.joint_position = curve_js1[i]
        wp.time_from_start = 5*lam[i]
        waypoints.append(wp)


    traj1 = JointTrajectory()
    traj1.joint_names = joint_names
    traj1.waypoints = waypoints

    robot1_obj.speed_ratio = 1

    traj1_gen = robot1_obj.execute_trajectory(traj1)

    ########################################################################
    robot2_obj = RRN.ConnectService(url[robot2])

    robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", robot2_obj)

    joint_names = [j.joint_identifier.name for j in robot2_obj.robot_info.joint_info]

    halt_mode = robot_const["RobotCommandMode"]["halt"]
    trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
    jog_mode = robot_const["RobotCommandMode"]["jog"]

    JointTrajectoryWaypoint = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectoryWaypoint",robot2_obj)
    JointTrajectory = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectory",robot2_obj)

    ###jog to start pose first
    robot2_obj.command_mode = halt_mode
    time.sleep(0.1)
    robot2_obj.command_mode =jog_mode
    robot2_obj.jog_freespace(curve_js2[0], np.ones(6), True)

    ###switch to traj mode
    robot2_obj.command_mode = halt_mode
    time.sleep(0.1)
    robot2_obj.command_mode = trajectory_mode


    state_w2 = robot2_obj.robot_state.Connect()

    state_w2.WaitInValueValid()

    waypoints = []



    for i in idx:
        wp = JointTrajectoryWaypoint()
        wp.joint_position = curve_js2[i]
        wp.time_from_start = 5*lam[i]
        waypoints.append(wp)


    traj2 = JointTrajectory()
    traj2.joint_names = joint_names
    traj2.waypoints = waypoints

    robot2_obj.speed_ratio = 1

    traj2_gen = robot2_obj.execute_trajectory(traj2)


    while (True):
        t = time.time()
        try:
            print(state_w1.InValue.joint_position)
            res = traj1_gen.AsyncNext(None,None)
            res = traj2_gen.AsyncNext(None,None)
            print(res)
        except RR.StopIterationException:
            break


if __name__ == "__main__":
    main()