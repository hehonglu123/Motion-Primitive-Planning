#!/usr/bin/env python3
from RobotRaconteur.Client import *
import time
import numpy as np
import sys

robot1 = RRN.ConnectService('rr+tcp://localhost:12222?service=robot')       
robot2 = RRN.ConnectService('rr+tcp://localhost:23333?service=robot')      


robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", robot1)
halt_mode = robot_const["RobotCommandMode"]["halt"]
jog_mode = robot_const["RobotCommandMode"]["jog"]
robot1.command_mode = halt_mode
time.sleep(0.1)
robot1.command_mode = jog_mode

robot2.command_mode = halt_mode
time.sleep(0.1)
robot2.command_mode = jog_mode


robot1.jog_freespace(np.zeros(6), np.ones(6), False)
robot2.jog_freespace(np.zeros(6), np.ones(6), False)
