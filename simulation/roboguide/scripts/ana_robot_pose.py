import numpy as np
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

motion='movel'
# motion='movec'

if motion == 'movel':
    mid_p=(start_p+end_p)/2
if motion == 'movec':
    mid_p = np.array([2500,0,1000])

save_dir='../data/robot_pose_test/'
ang=30

all_ori = [0,50]

for curve_ori in all_ori:

    if curve_ori == 0:
        all_pose_d_x = [0,-300,500]
        all_pose_d_y = [0,-1000,1000]
        all_pose = []
        for x in all_pose_d_x:
            for y in all_pose_d_y:
                p = np.array([x,y,0])
                all_pose.append(p)
    if curve_ori == 50:
        all_pose_d_x = [0]
        all_pose_d_y = [0,-1000,1000]
        all_pose = []
        for x in all_pose_d_x:
            for y in all_pose_d_y:
                p = np.array([x,y,0])
                all_pose.append(p)
    if curve_ori == 90:
        all_pose_d_x = [0]
        all_pose_d_y = [0,-1000,1000]
        all_pose = []
        for x in all_pose_d_x:
            for y in all_pose_d_y:
                p = np.array([x,y,0])
                all_pose.append(p)

    for curve_pose in all_pose:

        this_case = motion+'_ori'+str(curve_ori)+'_x'+str(curve_pose[0])+'_y'+str(curve_pose[1])
        print(this_case)

        try:
            with open(save_dir+'Curve_js_'+this_case+'.npy','rb') as f:
                curve_js = np.load(f)
        except OSError as e:
            print("No solution")
            continue 
            
        curve=[]
        curve_normal=[]
        for q in curve_js:
            this_pose=robot.fwd(q)
            curve.append(this_pose.p)
            curve_normal.append(this_pose.R[:,-1])
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

        lamdot_act=calc_lamdot(np.delete(curve_exe_js,dont_show_id,axis=0),lam_exec,robot,1)
        error,angle_error=calc_all_error_w_normal(curve_exe,curve,curve_exe_R[:,:,-1],curve_normal)

        # print(np.degrees(angle_error))
        print(np.max(error),'mm')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(lam_exec[1:],act_speed, 'g-', label='Speed')
        ax2.plot(lam_exec, error, 'b-',label='Cartesian Error')
        ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')
        ax1.legend(loc=0)
        ax2.legend(loc=0)
        ax1.set_xlabel('lambda (mm)')
        ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
        ax2.set_ylabel('Error (mm or deg)', color='b')
        ax2.set_ylim(0,1)
        plt.title('Error vs Lambda '+this_case)
        # plt.show()
        plt.savefig(save_dir+'error_speed_'+this_case+'.png')
        # exit()

        with open(save_dir+'error_'+this_case+'.npy','wb') as f:
            np.save(f,error)
        with open(save_dir+'normal_error_'+this_case+'.npy','wb') as f:
            np.save(f,angle_error)
        with open(save_dir+'speed_'+this_case+'.npy','wb') as f:
            np.save(f,act_speed)