from math import ceil, radians, floor, degrees
import numpy as np
from pandas import read_csv

from general_robotics_toolbox import *
import sys

# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
from lambda_calc import *
from error_check import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

# define m900ia
robot=m710ic(d=50)
client = FANUCClient()
utool_num = 2

all_objtype=['wood','blade']
# all_objtype=['blade']
# all_objtype=['wood']

thresholds=[0.1,0.2,0.5,0.9]
# thresholds=[1]

for obj_type in all_objtype:

    # obj_type='wood'
    # obj_type='blade'
    print(obj_type)
    
    data_dir='../data/baseline_m710ic/'+obj_type+'/'

    robot=m710ic(d=50)
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    curve = np.array(curve)
    curve_normal = curve[:,3:]
    curve = curve[:,:3]
    ###read actual curve
    # curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values
    # curve_js=np.array(curve_js)

    for threshold in thresholds:
        print(obj_type+' '+str(threshold))
        col_names=['','breakpoints','primitives','q1', 'q2', 'q3','q4', 'q5', 'q6'] 
        data = read_csv(data_dir+str(threshold)+"/command.csv", names=col_names)
        target_q1=data['q1'].tolist()[1:]
        target_q2=data['q2'].tolist()[1:]
        target_q3=data['q3'].tolist()[1:]
        target_q4=data['q4'].tolist()[1:]
        target_q5=data['q5'].tolist()[1:]
        target_q6=data['q6'].tolist()[1:]
        target_joints = np.vstack((target_q1,target_q2,target_q3,target_q4,target_q5,target_q6)).T.astype(float)

        # print(target_joints)
        # exit()

        speed_up=2000
        speed_low=0
        speed=(speed_up+speed_low)/2
        speed_found=False
        zone=100
        while True:
            speed=int(speed)
            # tp program
            # move to start
            robt1 = joint2robtarget(target_joints[0],robot,1,1,utool_num)
            robt2 = joint2robtarget(target_joints[1],robot,1,1,utool_num)
            start_direction=(robt1.trans-robt2.trans)/np.linalg.norm(robt1.trans-robt2.trans)
            robtend = joint2robtarget(target_joints[-1],robot,1,1,utool_num)
            robtend_1 = joint2robtarget(target_joints[-2],robot,1,1,utool_num)
            end_direction=(robtend.trans-robtend_1.trans)/np.linalg.norm(robtend.trans-robtend_1.trans)

            tp_pre = TPMotionProgram()
            j0 = jointtarget(1,1,utool_num,np.degrees(target_joints[0]),[0]*6)
            tp_pre.moveJ(j0,50,'%',-1)
            robt_start = joint2robtarget(target_joints[0],robot,1,1,utool_num)
            robt_start.trans[0] = robt_start.trans[0]+start_direction[0]*100
            robt_start.trans[1] = robt_start.trans[1]+start_direction[1]*100
            robt_start.trans[2] = robt_start.trans[2]+start_direction[2]*100
            # tp_pre.moveL(robt_start,5,'mmsec',-1)
            tp_pre.moveL(robt_start,500,'mmsec',-1)
            client.execute_motion_program(tp_pre)
            
            tp = TPMotionProgram()
            for target in target_joints:
                robt = joint2robtarget(target,robot,1,1,utool_num)
                # robt = jointtarget(1,1,utool_num,np.degrees(target),[0]*6)
                tp.moveL(robt,speed,'mmsec',zone)
            robt_end = joint2robtarget(target_joints[-1],robot,1,1,utool_num)
            robt_end.trans[0] = robt_end.trans[0]+end_direction[0]*100
            robt_end.trans[1] = robt_end.trans[1]+end_direction[1]*100
            robt_end.trans[2] = robt_end.trans[2]+end_direction[2]*100
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
        
        lamdot_act=calc_lamdot(curve_exe_js_act,lam_exec,robot,1)
        with open(data_dir+str(threshold)+'/error.npy','wb') as f:
            np.save(f,error)
        with open(data_dir+str(threshold)+'/normal_error.npy','wb') as f:
            np.save(f,angle_error)
        with open(data_dir+str(threshold)+'/speed.npy','wb') as f:
            np.save(f,act_speed)
        
        print(threshold,": Speed:",speed,',Max Error:',error_max,',Angle Error:',angle_error_max)
