# Copyright (c) 2020, Rensselaer Polytechnic Institute, Wason Technology LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Rensselaer Polytechnic Institute, nor Wason 
#       Technology LLC, nor the names of its contributors may be used to 
#       endorse or promote products derived from this software without 
#       specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE

import os
import numpy as np
# from urdf_parser_py.urdf import URDF
# import rospkg
import general_robotics_toolbox as rox
from urdf import robot_from_xml_string
import argparse
import yaml

def _rpy_to_rot(rpy):
    return rox.rot([0,0,1],rpy[2]).dot(rox.rot([0,1,0],rpy[1]))\
        .dot(rox.rot([1,0,0],rpy[0]))

def _vec_to_dict(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

def _rot_to_quaternion_dict(R):
    q = rox.R2q(R)
    return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}

def _pose_to_dict(R,p):
    return {"orientation": _rot_to_quaternion_dict(R), "position": _vec_to_dict(p)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=argparse.FileType('rb'))
    parser.add_argument("--root_link", type=str, help="root link of chain")
    parser.add_argument("--tip_link", type=str, help="tip link of chain")
    args = parser.parse_args()

    with args.file as f:
        file_str = f.read()

    #print file_str
    #print ("root_link: " + str(args.root_link))
    #print ("tip_link: " + str(args.tip_link))

    robot = robot_from_xml_string(file_str, args.root_link, args.tip_link)
    
    robot_info = {}

    joint_info = []
    for i in range(len(robot.joint_type)):
        joint_info_i = {}
        joint_info_i["joint_identifier"] = robot.joint_names[i]
        if robot.joint_type[i] == 0:
            joint_info_i["joint_type"] = "revolute"
            joint_info_i["default_units"] = "radian"
            joint_info_i["default_effort_units"] = "newton_meter"
        elif robot.joint_type[i] == 1:
            joint_info_i["joint_type"] = "prismatic"
            joint_info_i["default_units"] = "meter"
            joint_info_i["default_effort_units"] = "newton"
        else:
            assert False, "Invalid joint type for conversion"        
        joint_limits = {}
        if robot.joint_lower_limit is not None:
            joint_limits["lower"] = float(robot.joint_lower_limit[i])
        if robot.joint_upper_limit is not None:
            joint_limits["upper"] = float(robot.joint_upper_limit[i])
        if robot.joint_vel_limit is not None:
            joint_limits["velocity"] = float(robot.joint_vel_limit[i])
        if robot.joint_acc_limit is not None:
            joint_limits["acceleration"] = float(robot.joint_acc_limit[i])
        if hasattr(robot, "joint_effort_limit") and robot.joint_effort_limit is not None:
            joint_limits["effort"] = float(robot.joint_effort_limit[i])

        if (len(joint_limits) > 0):
            joint_info_i["joint_limits"] = joint_limits

        joint_info_i["passive"] = False
        
        joint_info.append(joint_info_i)

    robot_info["joint_info"] = joint_info

    kin_chain = {}
    kin_chain["kin_chain_identifier"] = "robot_arm"
    kin_chain["H"] = [_vec_to_dict(h) for h in robot.H.T]
    kin_chain["P"] = [_vec_to_dict(p) for p in robot.P.T]
    if robot.M is not None:
        link_inertias = []
        for M in robot.M:
            m = float(np.trace(M[3:6,3:6])/3.0)
            com = rox.invhat(M[0:3,3:6])/m
            I = M[0:3,0:3] - (m * np.inner(com,com) * np.eye(3) - m * np.outer(com,com))

            link_inertia = {}
            link_inertia["m"] = m
            link_inertia["com"] = _vec_to_dict(com)
            link_inertia["ixx"] = float(I[0,0])
            link_inertia["ixy"] = float(I[0,1])
            link_inertia["ixz"] = float(I[0,2])
            link_inertia["iyy"] = float(I[1,1])
            link_inertia["iyz"] = float(I[1,2])
            link_inertia["izz"] = float(I[2,2])

            link_inertias.append(link_inertia)
        
        kin_chain["link_inertias"] = link_inertias
    kin_chain["joint_numbers"] = range(len(robot.joint_type))
    kin_chain["flange_pose"] = _pose_to_dict(robot.R_tool, robot.p_tool)
    kin_chain["flange_identifier"] = robot.tip_link_name

    robot_info["chains"] = [kin_chain]

    print(yaml.dump(robot_info))

if __name__ == "__main__":
    main()
