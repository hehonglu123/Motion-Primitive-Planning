from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame
from math import sin,cos,ceil,floor
from copy import deepcopy
from pathlib import Path

from general_robotics_toolbox import *
import sys
from matplotlib import pyplot as plt

sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
from fanuc_motion_program_exec_client import *

def result_ana(curve_exe_js):
    act_speed=[0]
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
        last_cont = False

    curve_exe=np.array(curve_exe)
    curve_exe_R=np.array(curve_exe_R)
    curve_exe_js_act=np.array(curve_exe_js_act)
    # lamdot_act=calc_lamdot(curve_exe_js_act,lam_exec,robot,1)
    error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve_normal)

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
    act_speed_cut=act_speed[start_id:end_id]
    print("Ave Speed:",np.mean(act_speed_cut),'Max Error:',np.max(error),"Min Speed:",np.min(act_speed_cut))
    
    # return

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(lam_exec,act_speed, 'g-', label='Speed')
    ax2.plot(lam_exec, error, 'b-',label='Error')
    ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')
    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    plt.title("Execution Result (Speed/Error/Normal Error v.s. Lambda)")
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    plt.show()
    # plt.savefig(data_dir+'error_speed_'+case_file_name+'.png')
    # plt.clf()

    curve_plan = []
    for i in range(len(curve_js_plan)):
        this_q = curve_js_plan[i]
        robot_pose=robot.fwd(this_q)
        curve_plan.append(robot_pose.p)
    curve_plan=np.array(curve_plan)

    plt.plot(curve[:,0],curve[:,1])
    plt.plot(curve_plan[:,0],curve_plan[:,1])
    plt.plot(curve_exe[:,0],curve_exe[:,1])
    plt.axis('equal')
    plt.show()

    ###plot original curve
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
    ax.scatter3D(curve_plan[:,0], curve_plan[:,1],curve_plan[:,2], 'blue')
    #plot execution curve
    ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='execution')
    plt.show()

    return

    with open(data_dir+'error_'+case_file_name+'.npy','wb') as f:
        np.save(f,error)
    with open(data_dir+'normal_error_'+case_file_name+'.npy','wb') as f:
        np.save(f,angle_error)
    with open(data_dir+'speed_'+case_file_name+'.npy','wb') as f:
        np.save(f,act_speed)
    with open(data_dir+'lambda_'+case_file_name+'.npy','wb') as f:
        np.save(f,lam_exec)

# generate curve with linear orientation
# robot=abb6640(d=50)
robot=m710ic(d=50)


client = FANUCClient()
utool_num = 2

zone=100
speed=100
# tolerance=1
tolerance=0.5

try:
    with open('../Curve_js_m710ic.npy','rb') as f:
        curve_js = np.load(f)
    with open('../Curve_in_base_frame_m710ic.npy','rb') as f:
        curve = np.load(f)
    with open('../Curve_R_in_base_frame_m710ic.npy','rb') as f:
        R_all = np.load(f)
        curve_normal=R_all[:,:,-1]
except OSError as e:
    print("No curve file. Generating curve file...")
    ###generate a continuous arc, with linear orientation
    start_p = np.array([1700,500, 600])
    end_p = np.array([1700, -500, 600])
    mid_p=(start_p+end_p)/2

    ##### create curve #####
    ###start rotation by 'ang' deg
    k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
    k=k/np.linalg.norm(k)
    theta=np.radians(30)

    R=rot(k,theta)
    new_vec=R@(end_p-mid_p)
    new_end_p=mid_p+new_vec

    ###calculate lambda
    lam1_f=np.linalg.norm(mid_p-start_p)
    lam1=np.linspace(0,lam1_f,num=25001)
    lam_f=lam1_f+np.linalg.norm(mid_p-new_end_p)
    lam2=np.linspace(lam1_f,lam_f,num=25001)

    lam=np.hstack((lam1,lam2[1:]))

    #generate linear segment
    a1,b1,c1=lineFromPoints([lam1[0],start_p[0]],[lam1[-1],mid_p[0]])
    a2,b2,c2=lineFromPoints([lam1[0],start_p[1]],[lam1[-1],mid_p[1]])
    a3,b3,c3=lineFromPoints([lam1[0],start_p[2]],[lam1[-1],mid_p[2]])
    line1=np.vstack(((-a1*lam1-c1)/b1,(-a2*lam1-c2)/b2,(-a3*lam1-c3)/b3)).T

    a1,b1,c1=lineFromPoints([lam2[0],mid_p[0]],[lam2[-1],new_end_p[0]])
    a2,b2,c2=lineFromPoints([lam2[0],mid_p[1]],[lam2[-1],new_end_p[1]])
    a3,b3,c3=lineFromPoints([lam2[0],mid_p[2]],[lam2[-1],new_end_p[2]])
    line2=np.vstack(((-a1*lam2-c1)/b1,(-a2*lam2-c2)/b2,(-a3*lam2-c3)/b3)).T

    curve=np.vstack((line1,line2[1:]))

    R_init=Ry(np.radians(135))
    R_end=Ry(np.radians(90))
    # interpolate orientation
    R_all=[R_init]
    k,theta=R2rot(np.dot(R_end,R_init.T))
    curve_normal=[R_init[:,-1]]
    # theta=np.pi/4 #force 45deg change
    for i in range(1,len(curve)):
        angle=theta*i/(len(curve)-1)
        R_temp=rot(k,angle)
        R_all.append(np.dot(R_temp,R_init))
        curve_normal.append(R_all[-1][:,-1])
    curve_normal=np.array(curve_normal)

    q_init=robot.inv(start_p,R_init)[3]
    #solve inv kin

    curve_js=[q_init]
    for i in range(1,len(curve)):
        q_all=np.array(robot.inv(curve[i],R_all[i]))
        ###choose inv_kin closest to previous joints
        curve_js.append(unwrapped_angle_check(curve_js[-1],q_all))

    curve_js=np.array(curve_js)
    R_all=np.array(R_all)
    with open('../Curve_js_m710ic.npy','wb') as f:
        np.save(f,curve_js)
    with open('../Curve_in_base_frame_m710ic.npy','wb') as f:
        np.save(f,curve)
    with open('../Curve_R_in_base_frame_m710ic.npy','wb') as f:
        np.save(f,R_all)

total_seg = 200
step = int((len(curve_js)-1)/total_seg)
curve_js_plan = curve_js[::step]

tp_pre = TPMotionProgram()
j0=joint2robtarget(curve_js_plan[0],robot,1,1,utool_num)
tp_pre.moveJ(j0,50,'%',-1)
tp_pre.moveJ(j0,5,'%',-1)
client.execute_motion_program(tp_pre)

tp = TPMotionProgram()
for i in range(1,len(curve_js_plan)):
    robt=joint2robtarget(curve_js_plan[i],robot,1,1,utool_num)
    if i==len(curve_js_plan)-1:
        tp.moveL(robt,speed,'mmsec',-1)
    else:
        tp.moveL(robt,speed,'mmsec',zone)
# execute 
res = client.execute_motion_program(tp)
# Write log csv to file
with open("data/curve_js_"+str(total_seg)+"_exe.csv","wb") as f:
    f.write(res)

# read execution
curve_js_exe = read_csv("data/curve_js_"+str(total_seg)+"_exe.csv",header=None).values[1:]
curve_js_exe=np.array(curve_js_exe).astype(float)
timestamp = curve_js_exe[:,0]*1e-3
curve_js_exe = np.radians(curve_js_exe[:,1:])

result_ana(curve_js_exe)