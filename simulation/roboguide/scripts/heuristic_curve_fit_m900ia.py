from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame
from math import sin,cos
from copy import deepcopy

from general_robotics_toolbox import *
import sys
from matplotlib import pyplot as plt

from heuristic_curve_fit_func import *

sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../../../circular_fit')
from toolbox_circular_fit import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

def norm_vec(v):
    return v/np.linalg.norm(v)

# generate curve with linear orientation
# robot=abb6640(d=50)
robot=m900ia(d=50)
###generate a continuous arc, with linear orientation
start_p = np.array([2200,500, 1000])
end_p = np.array([2200, -500, 1000])
mid_p=(start_p+end_p)/2

data_dir = '../data/heuristic_curve_fit/'

client = FANUCClient()
utool_num = 2

ang=30

##### create curve #####
###start rotation by 'ang' deg
k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
k=k/np.linalg.norm(k)
theta=np.radians(ang)

R=rot(k,theta)
new_vec=R@(end_p-mid_p)
new_end_p=mid_p+new_vec

###calculate lambda
lam1_f=np.linalg.norm(mid_p-start_p)
lam1=np.linspace(0,lam1_f,num=25001)
lam_f=lam1_f+np.linalg.norm(mid_p-new_end_p)
lam2=np.linspace(lam1_f,lam_f,num=25001)

lam=np.hstack((lam1,lam2[1:]))

try:
    with open(data_dir+'Curve_js.npy','rb') as f:
        curve_js = np.load(f)
    with open(data_dir+'Curve_in_base_frame.npy','rb') as f:
        curve = np.load(f)
    with open(data_dir+'Curve_R_in_base_frame.npy','rb') as f:
        R_all = np.load(f)
        curve_normal=R_all[:,:,-1]
except OSError as e:

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

    q_init=robot.inv(start_p,R_init)[1]
    #solve inv kin

    curve_js=[q_init]
    for i in range(1,len(curve)):
        q_all=np.array(robot.inv(curve[i],R_all[-1]))
        ###choose inv_kin closest to previous joints
        curve_js.append(unwrapped_angle_check(curve_js[-1],q_all))

    curve_js=np.array(curve_js)
    R_all=np.array(R_all)
    with open(data_dir+'Curve_js.npy','wb') as f:
        np.save(f,curve_js)
    with open(data_dir+'Curve_in_base_frame.npy','wb') as f:
        np.save(f,curve)
    with open(data_dir+'Curve_R_in_base_frame.npy','wb') as f:
        np.save(f,R_all)


zone=100
tolerance=1

this_case='do_no_thing'



# bisection search robot speed
speed_up=2000
speed_low=0
speed=(speed_up+speed_low)/2
speed_found=False
zone=100
while True:
    speed=int(speed)
    # tp program
    # move to start
    tp_pre = TPMotionProgram()
    j0 = jointtarget(1,1,utool_num,np.degrees(target_joints[0]),[0]*6)
    tp_pre.moveJ(j0,50,'%',-1)
    robt_start = joint2robtarget(target_joints[0],robot,1,1,utool_num)
    # tp_pre.moveL(robt_start,5,'mmsec',-1)
    tp_pre.moveL(robt_start,500,'mmsec',-1)
    client.execute_motion_program(tp_pre)
    
    tp = TPMotionProgram()
    for target in target_joints[1:-1]:
        robt = joint2robtarget(target,robot,1,1,utool_num)
        # robt = jointtarget(1,1,utool_num,np.degrees(target),[0]*6)
        tp.moveL(robt,speed,'mmsec',zone)
    robt_end = joint2robtarget(target_joints[-1],robot,1,1,utool_num)
    tp.moveL(robt_end,speed,'mmsec',-1)

    # execute 
    res = client.execute_motion_program(tp)

    # Write log csv to file
    with open(data_dir+str(threshold)+"/curve_js_exe.csv","wb") as f:
        f.write(res)
    
    # the executed in Joint space (FANUC m710ic)
    col_names=['timestamp','J1', 'J2','J3', 'J4', 'J5', 'J6'] 
    # data=read_csv(data_dir+'movel_3_'+str(int(h*10))+'.csv',names=col_names)
    data=read_csv(data_dir+str(threshold)+"/curve_js_exe.csv",names=col_names)
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
        last_cont = False

    curve_exe=np.array(curve_exe)
    curve_exe_R=np.array(curve_exe_R)
    curve_exe_js_act=np.array(curve_exe_js_act)
    # lamdot_act=calc_lamdot(curve_exe_js_act,lam_exec,robot,1)
    error,angle_error=calc_all_error_w_normal(curve_exe,curve,curve_exe_R[:,:,-1],curve_normal)
    
    start_id = 10
    end_id = -10

    while True:
        if error[start_id+1]>=error[start_id]:
            break
        start_id+=1
    while True:
        if error[end_id-1]>=error[end_id]:
            break
        end_id-=1
    error=error[start_id:end_id+1]
    angle_error=angle_error[start_id:end_id+1]
    act_speed=act_speed[start_id-1:end_id]
    
    error_max = np.max(error)
    angle_error_max = np.max(angle_error)

    if speed_found:
        break

    
    print('Speed:',speed,',Error:',error_max,',Angle Error:',degrees(angle_error_max))
    if error_max > 1 or angle_error_max > radians(3):
        speed_up=speed
        speed=ceil((speed_up+speed_low)/2.)
    else:
        speed_low=speed
        speed=floor((speed_up+speed_low)/2.)
    if speed==speed_low:
        break
    elif speed==speed_up:
        speed=speed_low
        speed_found=True