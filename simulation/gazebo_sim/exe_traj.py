from RobotRaconteur.Client import *
import time, argparse
import numpy as np
from pandas import *



def main():
    #Accept the names of the webcams and the nodename from command line
    parser = argparse.ArgumentParser(description="RR trajectory execution")
    parser.add_argument("--robot-name",type=str)
    args, _ = parser.parse_known_args()

    robot1="abb6640"
    robot2="abb1200"

    url={robot1:'rr+tcp://localhost:12222?service=robot',robot2:'rr+tcp://localhost:23333?service=robot'}
    filename={robot1:'arm1.csv',robot2:'arm2.csv'}

    ###read actual curve
    col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    # data = read_csv("trajectory/"+filename[args.robot_name], names=col_names)
    data = read_csv("trajectory/single_arm/curve_pose_opt/arm1.csv", names=col_names)

    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T



    robot = RRN.ConnectService(url[args.robot_name])

    robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", robot)

    joint_names = [j.joint_identifier.name for j in robot.robot_info.joint_info]

    halt_mode = robot_const["RobotCommandMode"]["halt"]
    trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
    jog_mode = robot_const["RobotCommandMode"]["jog"]

    JointTrajectoryWaypoint = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectoryWaypoint",robot)
    JointTrajectory = RRN.GetStructureType("com.robotraconteur.robotics.trajectory.JointTrajectory",robot)

    ###jog to start pose first
    robot.command_mode = halt_mode
    time.sleep(0.1)
    robot.command_mode =jog_mode
    robot.jog_freespace(curve_js[0], np.ones(6), True)

    ###switch to traj mode
    robot.command_mode = halt_mode
    time.sleep(0.1)
    robot.command_mode = trajectory_mode


    state_w1 = robot.robot_state.Connect()

    state_w1.WaitInValueValid()

    waypoints = []



    for i in range(len(curve_js)):
        wp = JointTrajectoryWaypoint()
        wp.joint_position = curve_js[i]
        wp.time_from_start = 2*i/len(curve_js)
        waypoints.append(wp)


    traj = JointTrajectory()
    traj.joint_names = joint_names
    traj.waypoints = waypoints

    robot.speed_ratio = 1

    traj_gen = robot.execute_trajectory(traj)

    print('len(traj.waypoints): ',len(traj.waypoints))

    while (True):
        t = time.time()

        robot_state = state_w1.InValue
        try:
            res = traj_gen.Next()
            print(res)
        except RR.StopIterationException:
            break

        print(hex(robot.robot_state.PeekInValue()[0].robot_state_flags))

if __name__ == "__main__":
    main()