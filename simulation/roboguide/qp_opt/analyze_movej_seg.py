from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame
from math import sin,cos,ceil,floor
from copy import deepcopy
from pathlib import Path

from general_robotics_toolbox import *
import sys
import os
from matplotlib import pyplot as plt

sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../../../circular_fit')
from toolbox_circular_fit import *

robot=m900ia(d=50)
data_dir = 'data_movej_bisec_seg/'

all_total_seg=[50,500]

with open(data_dir+'Curve_js.npy','rb') as f:
    curve_js = np.load(f)
with open(data_dir+'Curve_in_base_frame.npy','rb') as f:
    curve = np.load(f)
with open(data_dir+'Curve_R_in_base_frame.npy','rb') as f:
    R_all = np.load(f)
    curve_normal=R_all[:,:,-1]

# the executed result
with open(data_dir+'error.npy','rb') as f:
    error = np.load(f)
with open(data_dir+'normal_error.npy','rb') as f:
    angle_error = np.load(f)
with open(data_dir+'speed.npy','rb') as f:
    act_speed = np.load(f)
with open(data_dir+'lambda.npy','rb') as f:
    lam_exec = np.load(f)

# act_speed = np.append(0,act_speed)
poly = np.poly1d(np.polyfit(lam_exec,act_speed,deg=40))
poly_der = np.polyder(poly)
# fit=poly(lam_exec[1:])     
start_id=10
end_id=-10
while True:
    if poly_der(lam_exec[start_id]) < 0.5:
        break
    start_id+=1
while True:
    if poly_der(lam_exec[end_id]) > -0.5:
        break
    end_id -= 1

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(lam_exec,act_speed, 'g-', label='Speed')
# ax1.plot(lam_exec[start_id:end_id],act_speed[start_id:end_id], 'r-', label='Speed')
ax2.plot(lam_exec, error, 'b-',label='Error')
# ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Execution Result ")
ax1.legend(loc=6)
ax2.legend(loc=7)
# plt.show()
plt.savefig(data_dir+'speed_error.png')
plt.clf()

error=error[start_id:end_id]
angle_error=angle_error[start_id:end_id]
act_speed=act_speed[start_id:end_id]
lam_exec=lam_exec[start_id:end_id]



print("Max Speed:",np.max(act_speed),"Min Speed:",np.min(act_speed),"Ave Speed:",np.mean(act_speed),"Speed Std:",np.std(act_speed),"Max Min %:",(np.max(act_speed)-np.min(act_speed))*100/np.max(act_speed))
print("Ave Err:",np.mean(error),"Max Err:",np.max(error),"Min Err:",np.min(error),"Err Std:",np.std(error))
print("Norm Ave Err:",np.mean(angle_error),"Norm Max Err:",np.max(angle_error),"Norm Min Err:",np.min(angle_error),"Norm Err Std:",np.std(angle_error))
print("Norm Ave Err:",degrees(np.mean(angle_error)),"Norm Max Err:",degrees(np.max(angle_error)),"Norm Min Err:",degrees(np.min(angle_error)),"Norm Err Std:",degrees(np.std(angle_error)))
print("======================")

# plot execution
# the executed in Joint space (FANUC m710ic)
col_names=['timestamp','J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data=read_csv(data_dir+"curve_js_exe.csv",names=col_names)
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]
curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float))
timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)*1e-3 # from msec to sec

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

    robot_pose=robot.fwd(this_q)
    curve_exe.append(robot_pose.p)
    curve_exe_R.append(robot_pose.R)
    curve_exe_js_act.append(this_q)
    last_cont = False

curve_exe=np.array(curve_exe)
curve_exe_R=np.array(curve_exe_R)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='Motion Cmd')
#plot execution curve
ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='Executed Motion')
ax.view_init(elev=40, azim=-145)
ax.set_title('Cartesian Interpolation using Motion Cmd')
ax.set_xlabel('x-axis (mm)')
ax.set_ylabel('y-axis (mm)')
ax.set_zlabel('z-axis (mm)')
plt.show()