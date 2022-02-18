#!/bin/sh
cd ~/GazeboModelRobotRaconteurDriver-2021-04-18
# RR URL: rr+tcp://localhost:12222?service=robot
dotnet GazeboModelRobotRaconteurDriver.dll --gazebo-url=rr+tcp://localhost:11346/?service=GazeboServer --robotraconteur-tcp-port=12222 --robotraconteur-nodename=fanuc_m900ia --model-name=fanuc_m900ia --robot-info-file=/home/eric/Motion-Primitive-Planning/simulation/gazebo_sim/config/fanuc_m900ia.yml
