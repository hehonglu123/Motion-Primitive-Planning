#!/bin/sh
cd ~/GazeboModelRobotRaconteurDriver-2021-04-18
# RR URL: rr+tcp://localhost:23333?service=robot
dotnet GazeboModelRobotRaconteurDriver.dll --gazebo-url=rr+tcp://localhost:11346/?service=GazeboServer --robotraconteur-tcp-port=23333 --robotraconteur-nodename=abb_6650s2 --model-name=abb6650s2 --robot-info-file=/mnt/c/Users/hehon/Desktop/robodk/Motion-Primitive-Planning/simulation/gazebo_sim/config/6650s2.yml
