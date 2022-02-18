#!/bin/sh
cd ~/GazeboModelRobotRaconteurDriver-2021-04-18
# RR URL: rr+tcp://localhost:12222?service=robot
dotnet GazeboModelRobotRaconteurDriver.dll --gazebo-url=rr+tcp://localhost:11346/?service=GazeboServer --robotraconteur-tcp-port=12222 --robotraconteur-nodename=abb_6640 --model-name=abb6640 --robot-info-file=/home/eric/Motion-Primitive-Planning/simulation/gazebo_sim/config/abb6640.yml
