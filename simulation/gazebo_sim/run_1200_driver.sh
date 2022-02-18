#!/bin/sh
cd ~/GazeboModelRobotRaconteurDriver-2021-04-18
# RR URL: rr+tcp://localhost:23333?service=robot
dotnet GazeboModelRobotRaconteurDriver.dll --gazebo-url=rr+tcp://localhost:11346/?service=GazeboServer --robotraconteur-tcp-port=23333 --robotraconteur-nodename=abb_1200 --model-name=abb1200 --robot-info-file=/home/eric/Motion-Primitive-Planning/simulation/gazebo_sim/config/abb1200.yml

# dotnet GazeboModelRobotRaconteurDriver.dll --gazebo-url=rr+local:///?nodeid=8a1cebc4-6400-4502-9176-21f27633537a&service=GazeboServer --robotraconteur-tcp-port=23333 --robotraconteur-nodename=abb_1200 --model-name=abb1200 --robot-info-file=/home/eric/Motion-Primitive-Planning/simulation/gazebo_sim/config/abb1200.yml
