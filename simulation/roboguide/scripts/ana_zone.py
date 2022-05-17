import numpy as np
from numpy.linalg import norm
from scipy.interpolate import UnivariateSpline, BPoly
from math import ceil
from copy import deepcopy
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv, read_excel
import sys
sys.path.append('../../../toolbox')
from abb_motion_program_exec_client import *
from robots_def import *
from lambda_calc import *
from error_check import *

def norm_vec(v):
    return v/np.linalg.norm(v)

robot = m900ia(d=50)

start_p = np.array([2200,500, 1000])
end_p = np.array([2200, -500, 1000])
mid_p=(start_p+end_p)/2

save_dir='../data/robot_pose_test/'
ang=30

this_case = 'movel_ori0_x0_y0'
with open(save_dir+'Curve_js_'+this_case+'.npy','rb') as f:
    curve_js = np.load(f)
curve_js=curve_js[::2]

curve=[]
curve_normal=[]
lam=[0]
i=0
for q in curve_js:
    this_pose=robot.fwd(q)
    curve.append(this_pose.p)
    curve_normal.append(this_pose.R[:,-1])
    if i>0:
        lam.append(lam[-1]+np.linalg.norm(curve[-1]-curve[-2]))
    i+=1
curve=np.array(curve)
curve_normal=np.array(curve_normal)

# the executed in Joint space (FANUC m900iA)
col_names=['timestamp','J1', 'J2','J3', 'J4', 'J5', 'J6'] 
# data=read_csv(data_dir+'movel_3_'+str(int(h*10))+'.csv',names=col_names)
data=read_csv(save_dir+'Curve_exec_'+this_case+'.csv',names=col_names)
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]
curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float))
timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)*1e-3 # from msec to sec

act_speed=[]
lam_exec=[0]
curve_exe=[]
curve_exe_R=[]
curve_exe_js_act=[]
dont_show_id=[]
last_cont = False
for i in range(len(curve_exe_js)):
    this_q = curve_exe_js[i]
    if i>5 and i<len(curve_exe_js)-5:
        # if the recording is not fast enough
        # then having to same logged joint angle
        # do interpolation for estimation
        if np.all(this_q==curve_exe_js[i+1]):
            dont_show_id=np.append(dont_show_id,i).astype(int)
            last_cont = True
            continue
            # this_q = (curve_exe_js[i-1]+curve_exe_js[i+1])/2
            # this_q=np.array([])
            # for j in range(6):
            #     poly_q=BPoly.from_derivatives([timestamp[i-1],timestamp[i+1]],\
            #                 [[curve_exe_js[i-1,j],(curve_exe_js[i-1,j]-curve_exe_js[i-1-1,j])/(timestamp[i-1]-timestamp[i-1-1])],\
            #                 [curve_exe_js[i+1,j],(curve_exe_js[i+1+1,j]-curve_exe_js[i+1,j])/(timestamp[i+1+1]-timestamp[i+1])]])
            #     this_q=np.append(this_q,poly_q(timestamp[i]))

    robot_pose=robot.fwd(this_q)
    curve_exe.append(robot_pose.p)
    curve_exe_R.append(robot_pose.R)
    curve_exe_js_act.append(this_q)
    if i>0:
        lam_exec.append(lam_exec[-1]+np.linalg.norm(curve_exe[-1]-curve_exe[-2]))
    try:
        if timestamp[-1]!=timestamp[-2]:
            if last_cont:
                timestep=timestamp[i]-timestamp[i-2]
            else:
                timestep=timestamp[i]-timestamp[i-1]
            act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
    except IndexError:
        pass
curve_exe=np.array(curve_exe)
curve_exe_R=np.array(curve_exe_R)
curve_exe_js_act=np.array(curve_exe_js_act)

###plot original curve
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
# breakpoints=np.array([0,int((len(curve)+1)/2),int(len(curve)-1)])
# ax.scatter3D(curve[breakpoints,0], curve[breakpoints,1],curve[breakpoints,2], 'blue')
# #plot execution curve
# ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='execution')
# plt.show()

# try some fit

# zone_dist=168
# zone_dist=170
zone_dist=170
lam_bp=lam[int((len(curve)+1)/2)]
blending_num = ceil(zone_dist/np.linalg.norm(curve[20]-curve[19]))
breakpoint_i = int((len(curve)+1)/2)
zone_start_i = int(breakpoint_i-blending_num)
zone_end_i = int(breakpoint_i+blending_num)
# print(zone_start_i)
# print(zone_end_i)
# print(zone_end_i-zone_start_i)
# zone_start_p = norm_vec(start_p-mid_p)*zone_dist+mid_p
# zone_end_p = norm_vec(end_p-mid_p)*zone_dist+mid_p

cart_blend=False
use_exec_lam=False

save_dir='../data/zone_ana/'
this_case='joint_blend_cmd'

if cart_blend:

    if not use_exec_lam:
        lam_end_origin=lam[zone_end_i]
        for round in range(1):
            curve_blend = deepcopy(curve)
            poly_x=BPoly.from_derivatives([lam[zone_start_i],lam[zone_end_i]],\
                            [[curve[zone_start_i,0],(curve[zone_start_i,0]-curve[zone_start_i-1,0])/(lam[zone_start_i]-lam[zone_start_i-1])],\
                            [curve[zone_end_i,0],(curve[zone_end_i+1,0]-curve[zone_end_i,0])/(lam[zone_end_i+1]-lam[zone_end_i])]])
            print(poly_x)
            zone_x_fit=poly_x(lam[zone_start_i:zone_end_i+1])
            poly_y=BPoly.from_derivatives([lam[zone_start_i],lam[zone_end_i]],\
                            [[curve[zone_start_i,1],(curve[zone_start_i,1]-curve[zone_start_i-1,1])/(lam[zone_start_i]-lam[zone_start_i-1])],\
                            [curve[zone_end_i,1],(curve[zone_end_i+1,1]-curve[zone_end_i,1])/(lam[zone_end_i+1]-lam[zone_end_i])]])
            zone_y_fit=poly_y(lam[zone_start_i:zone_end_i+1])
            curve_blend[zone_start_i:zone_end_i+1,0]=zone_x_fit
            curve_blend[zone_start_i:zone_end_i+1,1]=zone_y_fit

            lam_count=0
            for i in range(1,zone_end_i+1):
                lam_count+=np.linalg.norm(curve_blend[i]-curve_blend[i-1])
                lam[i]=lam_count
            print(lam[zone_end_i]-lam_end_origin)
            lam[zone_end_i+1:] = lam[zone_end_i+1:]+lam[zone_end_i]-lam_end_origin
    
    if use_exec_lam:
        curve_blend = deepcopy(curve_exe)
        zone_start_i = 0
        zone_end_i = len(curve_exe)-1
        while True:
            pd = norm(curve_exe[zone_start_i]-mid_p)
            if pd <= zone_dist:
                print(curve_exe[zone_start_i])
                break
            zone_start_i+=1
        while True:
            pd = norm(curve_exe[zone_end_i]-mid_p)
            if pd <= zone_dist:
                zone_end_i+=1
                break
            zone_end_i-=1
        poly_x=BPoly.from_derivatives([lam_exec[zone_start_i],lam_exec[zone_end_i]],\
                        [[curve_blend[zone_start_i,0],(curve_blend[zone_start_i,0]-curve_blend[zone_start_i-1,0])/(lam_exec[zone_start_i]-lam_exec[zone_start_i-1])],\
                        [curve_blend[zone_end_i,0],(curve_blend[zone_end_i+1,0]-curve_blend[zone_end_i,0])/(lam_exec[zone_end_i+1]-lam_exec[zone_end_i])]])
        poly_y=BPoly.from_derivatives([lam_exec[zone_start_i],lam_exec[zone_end_i]],\
                        [[curve_blend[zone_start_i,1],(curve_blend[zone_start_i,1]-curve_blend[zone_start_i-1,1])/(lam_exec[zone_start_i]-lam_exec[zone_start_i-1])],\
                        [curve_blend[zone_end_i,1],(curve_blend[zone_end_i+1,1]-curve_blend[zone_end_i,1])/(lam_exec[zone_end_i+1]-lam_exec[zone_end_i])]])
        zone_x_fit=poly_x(lam_exec[zone_start_i:zone_end_i])
        zone_y_fit=poly_y(lam_exec[zone_start_i:zone_end_i])
        curve_blend[zone_start_i:zone_end_i,0]=zone_x_fit
        curve_blend[zone_start_i:zone_end_i,1]=zone_y_fit
        curve_blend[:,2]=1000
else:
    print("blend joint")

    if not use_exec_lam:
        curve_js_blend = deepcopy(curve_js)
        curve_blend = deepcopy(curve)
        for j in range(6):
            poly_q=BPoly.from_derivatives([lam[zone_start_i],lam[zone_end_i]],\
                        [[curve_js_blend[zone_start_i,j],(curve_js_blend[zone_start_i,j]-curve_js_blend[zone_start_i-1,j])/(lam[zone_start_i]-lam[zone_start_i-1])],\
                        [curve_js_blend[zone_end_i,j],(curve_js_blend[zone_end_i+1,j]-curve_js_blend[zone_end_i,j])/(lam[zone_end_i+1]-lam[zone_end_i])]])
            curve_js_blend[zone_start_i:zone_end_i,j]=poly_q(lam[zone_start_i:zone_end_i])

    if use_exec_lam:
        curve_js_blend = deepcopy(curve_exe_js_act)
        curve_blend = deepcopy(curve_exe)
        # zone_start_i = np.searchsorted(lam_exec,lam_bp-zone_dist)-1
        # zone_end_i = np.searchsorted(lam_exec,lam_bp+zone_dist)
        zone_start_i = 0
        zone_end_i = len(curve_exe)-1
        while True:
            pd = norm(curve_exe[zone_start_i]-mid_p)
            if pd <= zone_dist:
                break
            zone_start_i+=1
        while True:
            pd = norm(curve_exe[zone_end_i]-mid_p)
            if pd <= zone_dist:
                break
            zone_end_i-=1
        for j in range(6):
            poly_q=BPoly.from_derivatives([lam_exec[zone_start_i],lam_exec[zone_end_i]],\
                        [[curve_js_blend[zone_start_i,j],(curve_js_blend[zone_start_i,j]-curve_js_blend[zone_start_i-1,j])/(lam_exec[zone_start_i]-lam_exec[zone_start_i-1])],\
                        [curve_js_blend[zone_end_i,j],(curve_js_blend[zone_end_i+1,j]-curve_js_blend[zone_end_i,j])/(lam_exec[zone_end_i+1]-lam_exec[zone_end_i])]])
            curve_js_blend[zone_start_i:zone_end_i,j]=poly_q(lam_exec[zone_start_i:zone_end_i])
    
    zone_x_fit=[]
    zone_y_fit=[]
    blending_curve=[]
    i=0
    for q in curve_js_blend[zone_start_i:zone_end_i]:
        robot_pose=robot.fwd(q)
        zone_x_fit.append(robot_pose.p[0])
        zone_y_fit.append(robot_pose.p[1])
        blending_curve.append(robot_pose.p)
        i+=1
    blending_curve=np.array(blending_curve)
    curve_blend[zone_start_i:zone_end_i]=blending_curve


# plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='Motion Cmd')
breakpoints=np.array([0,int((len(curve)+1)/2),int(len(curve)-1)])
# ax.scatter3D(curve[breakpoints,0], curve[breakpoints,1],curve[breakpoints,2], 'blue',label='(Start/End/Break) points')
# #plot execution curve
# ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='Executed Motion')
# ax.plot3D(curve_blend[zone_start_i:zone_end_i+1,0], curve_blend[zone_start_i:zone_end_i+1,1],curve_blend[zone_start_i:zone_end_i+1,2], 'blue',label='Interpolation')
# ax.view_init(elev=40, azim=-145)
# ax.set_title('Cartesian Interpolation using Motion Cmd')
# ax.set_xlabel('x-axis (mm)')
# ax.set_ylabel('y-axis (mm)')
# ax.set_zlabel('z-axis (mm)')
# ax.set_xlim(2200,2550)
# ax.set_ylim(-500,600)
# ax.set_zlim(999.4,1000.02)
# ax.legend()
# plt.show()
# plt.clf()

plt.plot(curve[:,1],curve[:,0], 'red',label='Motion Cmd')
plt.scatter(curve[breakpoints,1], curve[breakpoints,0],8,'royalblue',label='(Start/End/Break) points')
plt.plot(curve_exe[:,1],curve_exe[:,0], 'green',label='Executed Motion')
plt.plot(curve_blend[zone_start_i:zone_end_i,1],curve_blend[zone_start_i:zone_end_i,0], 'blue',label='Interpolation')
# plt.plot(zone_x_fit,zone_y_fit,'blue')
plt.axis('equal')
plt.title('Joint Interpolation using Motion Cmd (XY plane)')
plt.gca().invert_xaxis()
plt.xlim(173,-151)
plt.ylim(2175,2364)
plt.legend()
plt.xlabel('y-axis (mm)')
plt.ylabel('x-axis (mm)')
plt.show()

exit()

if cart_blend:
    if not use_exec_lam:
        plt.plot(lam,curve[:,0],'red')
        plt.plot(lam_exec,curve_exe[:,0],'green')
        plt.plot(lam[zone_start_i:zone_end_i+1],zone_x_fit,'blue')
        plt.title('x blend')
        plt.show()
        plt.plot(lam,curve[:,1],'red')
        plt.plot(lam_exec,curve_exe[:,1],'green')
        plt.plot(lam[zone_start_i:zone_end_i+1],zone_y_fit,'blue')
        plt.title('y blend')
        plt.show()
    else:
        plt.plot(lam,curve[:,0],'red')
        plt.plot(lam_exec,curve_exe[:,0],'green')
        plt.plot(lam_exec[zone_start_i:zone_end_i],zone_x_fit,'blue')
        plt.title('x blend')
        plt.show()
        plt.plot(lam,curve[:,1],'red')
        plt.plot(lam_exec,curve_exe[:,1],'green')
        plt.plot(lam_exec[zone_start_i:zone_end_i],zone_y_fit,'blue')
        plt.title('y blend')
        plt.show()
else:
    if not use_exec_lam:
        for i in range(6):
            plt.plot(lam,curve_js[:,i],'red')
            plt.plot(lam_exec,curve_exe_js_act[:,i],'green')
            plt.plot(lam[zone_start_i:zone_end_i],curve_js_blend[zone_start_i:zone_end_i,i],'blue')
            plt.title('joint '+str(i+1))
            plt.show()
    else:
        for i in range(6):
            plt.plot(lam,curve_js[:,i],'red')
            plt.plot(lam_exec,curve_exe_js_act[:,i],'green')
            plt.plot(lam_exec[zone_start_i:zone_end_i],curve_js_blend[zone_start_i:zone_end_i,i],'blue')
            plt.title('joint '+str(i+1))
            plt.show()




# with open(save_dir+'error_'+this_case+'.npy','wb') as f:
#     np.save(f,error)
# with open(save_dir+'normal_error_'+this_case+'.npy','wb') as f:
#     np.save(f,angle_error)
# with open(save_dir+'speed_'+this_case+'.npy','wb') as f:
#     np.save(f,act_speed)

exit()

plt.plot(lam,curve[:,0],'red')
plt.plot(lam_exec,curve_exe[:,0],'green')
plt.plot(lam[zone_start_i:zone_end_i],zone_x_fit,'blue')
plt.show()
plt.plot(lam,curve[:,1],'red')
plt.plot(lam_exec,curve_exe[:,1],'green')
plt.plot(lam[zone_start_i:zone_end_i],zone_y_fit,'blue')
plt.show()

