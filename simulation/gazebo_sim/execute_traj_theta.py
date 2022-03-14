from RobotRaconteur.Client import *
import time, argparse, sys
import numpy as np
from pandas import *

sys.path.append('../../toolbox')
from robots_def import *


def main():
    ###read actual curve
    col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv("trajectory/single_arm/all_theta_opt/arm1.csv", names=col_names)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    curve_js1=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

    robot=abb6640()
    p_prev=robot.fwd(curve_js1[0]).p
    ###find path length
    lam=[0]
    for i in range(len(curve_js1)-1):
        p_new=robot.fwd(curve_js1[i+1]).p
        lam.append(lam[-1]+np.linalg.norm(p_new-p_prev))
        p_prev=p_new
    ###normalize lam, 
    lam=np.array(lam)/lam[-1]


    robot1 = RRN.ConnectService('rr+tcp://localhost:12222?service=robot')

    robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", robot1)

    joint_names = [j.joint_identifier.name for j in robot1.robot_info.joint_info]

    halt_mode = robot_const["RobotCommandMode"]["halt"]
    trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
    jog_mode = robot_const["RobotCommandMode"]["jog"]

    JointTrajectoryWaypoint = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectoryWaypoint",robot1)
    JointTrajectory = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectory",robot1)

    ###jog to start pose first
    robot1.command_mode = halt_mode
    time.sleep(0.1)
    robot1.command_mode =jog_mode
    robot1.jog_freespace(curve_js1[0], np.ones(6), True)

    ###switch to traj mode
    robot1.command_mode = halt_mode
    time.sleep(0.1)
    robot1.command_mode = trajectory_mode


    state_w1 = robot1.robot_state.Connect()

    state_w1.WaitInValueValid()

    waypoints = []



    for i in range(len(curve_js1)):
        wp = JointTrajectoryWaypoint()
        wp.joint_position = curve_js1[i]
        wp.time_from_start = 3*lam[i]
        waypoints.append(wp)


    traj1 = JointTrajectory()
    traj1.joint_names = joint_names
    traj1.waypoints = waypoints

    robot1.speed_ratio = 1

    traj1_gen = robot1.execute_trajectory(traj1)


    while (True):
        t = time.time()
        try:
            res = traj1_gen.Next()
            print(res)
        except RR.StopIterationException:
            break


if __name__ == "__main__":
    main()