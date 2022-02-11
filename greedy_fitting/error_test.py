import csv
from matplotlib.pyplot import colormaps, contour
from pandas import *
import numpy as np
import sys
import matplotlib.pyplot as plt

# from toolbox.robot_def import fwd
sys.path.append('../toolbox')
from robots_def import *
import general_robotics_toolbox as rox

robot=abb6640()

# read Robotstudio logged data
with open("logged_joints/log_z1.csv","r") as f:
    rows = csv.reader(f, delimiter=',')

    log_results_dict = {}
    for col in rows:
        if len(log_results_dict) == 0:
            log_results_dict['timestamp']=[]
            log_results_dict['cmd_num']=[]
            log_results_dict['joint_angle']=[]
            continue
        log_results_dict['timestamp'].append(float(col[0]))
        log_results_dict['cmd_num'].append(int(col[1]))
        log_results_dict['joint_angle'].append(np.deg2rad(np.array([float(col[2]),float(col[3]),float(col[4]),float(col[5]),float(col[6]),float(col[7])])))
    stamps = log_results_dict['timestamp']
    cmd_num = log_results_dict['cmd_num']
    joint_angles = log_results_dict['joint_angle']

# read breakpoint data
with open("command_backproj_30000.csv","r") as f:
    rows = csv.reader(f, delimiter=',')

    command_dict = {}
    breakpoints = []
    breakpoints_out = []
    primitives = []
    for col in rows:
        if len(command_dict) == 0:
            command_dict['breakpoints']=[]
            command_dict['breakpoints_out']=[]
            command_dict['primitives']=[]
            continue
        command_dict['breakpoints'].append(int(col[0]))
        command_dict['breakpoints_out'].append(int(col[1]))
        command_dict['primitives'].append(col[2])
    breakpoints = command_dict['breakpoints']
    breakpoints_out = command_dict['breakpoints_out']
    primitives = command_dict['primitives']

# read curve
col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../data/from_ge/Curve_in_base_frame.csv", names=col_names)
test_length = breakpoints[-1]+1
curve_x=data['X'].tolist()[:test_length]
curve_y=data['Y'].tolist()[:test_length]
curve_z=data['Z'].tolist()[:test_length]
curve_direction_x=data['direction_x'].tolist()[:test_length]
curve_direction_y=data['direction_y'].tolist()[:test_length]
curve_direction_z=data['direction_z'].tolist()[:test_length]
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T

# print(curve)

plt.figure()
ax = plt.axes(projection='3d')
colormap=['red','orange','yellow','green','cyan','blue','purple']

stand_off = 50
curve_backproj = curve-stand_off*curve_normal

all_exec_path = np.array([[0,0,0]])
all_exec_path_proj = np.array([[0,0,0]])
all_error = []
all_error_proj=[]
# skip first break point since it's moveJ to the initial point
for i in range(0,len(breakpoints)):

    # motion_seg = np.argwhere(np.array(cmd_num)==i).flatten() # the cum_num start with 1
    motion_seg = np.argwhere(np.array(cmd_num)==i+1).flatten()
    # print(motion_seg)
    motion_start = motion_seg[0]
    motion_end = motion_seg[-1]

    # Curve
    # this_curve = curve[breakpoints[i-1]:breakpoints[i]]
    # this_curve_backproj=this_curve-stand_off*curve_normal[breakpoints[i-1]:breakpoints[i]]
    # this_curve_l = [0]
    # for curve_i in range(1,len(this_curve)):
    #     this_curve_l.append(this_curve_l[-1]+np.linalg.norm(this_curve[curve_i]-this_curve[curve_i-1]))
    # this_curve_l = np.array(this_curve_l)/this_curve_l[-1]
    
    # # Real robot (or in RobotStudio) execution path
    exec_path = []
    exec_path_proj = []
    exec_path_proj_l = [0]
    for mo_i in range(motion_start,motion_end+1):
        exec_tool_T = robot.fwd(joint_angles[mo_i])
        exec_path.append(exec_tool_T.p)
        exec_curve_T = exec_tool_T*rox.Transform(np.eye(3),[0,0,stand_off])
        exec_path_proj.append(exec_curve_T.p)
        if len(exec_path_proj)>1:
            exec_path_proj_l.append(exec_path_proj_l[-1]+np.linalg.norm(exec_path_proj[-1]-exec_path_proj[-2]))
    exec_path_proj_l = np.array(exec_path_proj_l)/exec_path_proj_l[-1]
    exec_path_proj = np.array(exec_path_proj)
    exec_path = np.array(exec_path)
    # print(this_curve_l)
    # print(exec_path_proj_l)
    # print(joint_angles[motion_start:motion_end+1])
    all_exec_path = np.vstack((all_exec_path,exec_path))
    all_exec_path_proj = np.vstack((all_exec_path_proj,exec_path_proj))

    # # error calculation
    # # associate the curve and execution path with the percentage of the path length
    # curve_error = [np.linalg.norm(exec_path_proj[0]-this_curve[0])]
    # backproj_error = [np.linalg.norm(exec_path[0]-this_curve_backproj[0])]
    # for exec_i in range(1,len(exec_path_proj_l)):
    #     curve_where = np.argmax(this_curve_l>=exec_path_proj_l[exec_i])
    #     prev_portion = (this_curve_l[curve_where]-exec_path_proj_l[exec_i])/(this_curve_l[curve_where]-this_curve_l[curve_where-1])
    #     post_portion = (exec_path_proj_l[exec_i]-this_curve_l[curve_where-1])/(this_curve_l[curve_where]-this_curve_l[curve_where-1])
        
    #     # do interpolation between two curve points
    #     associated_curve_point = prev_portion*this_curve[curve_where-1]+post_portion*this_curve[curve_where]
    #     associated_curve_backproj_point = prev_portion*this_curve_backproj[curve_where-1]+post_portion*this_curve_backproj[curve_where]

    #     # error calculation
    #     curve_error.append(np.linalg.norm(exec_path_proj[exec_i]-associated_curve_point))
    #     backproj_error.append(np.linalg.norm(exec_path[exec_i]-associated_curve_backproj_point))

    # print("Ave Proj Error:",np.mean(curve_error))
    # print("Max Proj Error:",np.max(curve_error))
    # all_error_proj = np.append(all_error_proj,curve_error)

    # print("Ave Error:",np.mean(backproj_error))
    # print("Max Error:",np.max(backproj_error))
    # print("=============================")
    # all_error = np.append(all_error,backproj_error)

    print(colormap[i])
    ax.scatter3D(exec_path[1:,0], exec_path[1:,1], exec_path[1:,2], color=colormap[i])

    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(this_curve[:,0], this_curve[:,1],this_curve[:,2], 'gray')
    # ax.scatter3D(exec_path_proj[:,0], exec_path_proj[:,1], exec_path_proj[:,2], c=exec_path_proj[:,2], cmap='Greens')
    # plt.show()

    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(this_curve_backproj[:,0], this_curve_backproj[:,1],this_curve_backproj[:,2], 'gray')
    # ax.scatter3D(exec_path[:,0], exec_path[:,1], exec_path[:,2], c=exec_path[:,2], cmap='Greens')
    # plt.show()

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
# ax.scatter3D(all_exec_path_proj[1:,0], all_exec_path_proj[1:,1], all_exec_path_proj[1:,2], cmap='Greens')
# # ax.axis('equal')
# plt.show()

# plt.figure()
# ax = plt.axes(projection='3d')
ax.plot3D(curve_backproj[:,0], curve_backproj[:,1],curve_backproj[:,2], 'gray')
# ax.scatter3D(all_exec_path[1:,0], all_exec_path[1:,1], all_exec_path[1:,2], cmap='Greens')
# ax.axis('equal')
plt.show()

print("Ave All Proj Error:",np.mean(all_error_proj))
print("Max All Proj Error:",np.max(all_error_proj))
print("Ave All Error:",np.mean(all_error))
print("Max All Error:",np.max(all_error))
print("=============================")