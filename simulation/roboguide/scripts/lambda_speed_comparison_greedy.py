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
sys.path.append('../../../greedy_fitting')
from greedy_poly import *

# move_type='movel'
# move_type='movej'

# data_dir="../data/slicing_compare_"+move_type+'/'
# data_dir="../data/greedy_poly/"
data_dir="../data/greedy_poly_offset/"
robot = m900ia(d=50)

x_offset=500
y_offset=500

# the original curve in Cartesian space
col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_x = np.array(curve_x)+x_offset
curve_y = np.array(curve_y)-y_offset
curve=np.vstack((curve_x, curve_y, curve_z)).T

max_error_mat=np.zeros((4,))
# ave_error_mat=np.zeros((4,4))
ave_speed_mat=np.zeros((4,))

all_zones=[25,50,75,100]
# all_zones=[100]

mat_i=0

for zone in all_zones:

    ### the curve executed
    col_names=['timestamp','J1', 'J2','J3', 'J4', 'J5', 'J6'] 
    data = read_csv(data_dir+"greedy_poly_"+str(zone)+".csv",names=col_names)
    # data = read_csv("log_test.csv",names=col_names)
    q1=data['J1'].tolist()[1:]
    q2=data['J2'].tolist()[1:]
    q3=data['J3'].tolist()[1:]
    q4=data['J4'].tolist()[1:]
    q5=data['J5'].tolist()[1:]
    q6=data['J6'].tolist()[1:]
    curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float))
    timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)*1e-3 # from msec to sec

    timestep=np.average(timestamp[1:]-timestamp[:-1])

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
                # linear interpolation
                # this_q=(curve_exe_js[i+1]+curve_exe_js[i-1])/2
                # # 3rd order interpolation
                # this_q=np.array([])
                # for qi in range(6):
                #     param = np.polyfit([timestamp[i-2],timestamp[i-1],timestamp[i+1],timestamp[i+2]],[curve_exe_js[i-2,qi],curve_exe_js[i-1,qi],\
                #                                 curve_exe_js[i+1,qi],curve_exe_js[i+2,qi]],3)
                #     this_q = np.append(this_q,np.polyval(param,timestamp[i]))
                # curve_exe_js[i]=this_q
                dont_show_id=np.append(dont_show_id,i).astype(int)
                last_cont = True
                continue
            # elif np.all(this_q==curve_exe_js[i-1]):
            #     dont_show_id=np.append(dont_show_id,i).astype(int)

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

    # lamdot_act=calc_lamdot(curve_exe_js,lam_exec,robot,1)
    lamdot_act=calc_lamdot(np.delete(curve_exe_js,dont_show_id,axis=0),lam_exec,robot,1)
    error_all=calc_all_error(curve_exe,curve)

    # plt.plot(lam_fit[0:-1:calc_ldot_step],lamdot_fit, label='Ldot Constraint')
    plt.plot(lam_exec,lamdot_act, label='Logged Joint Ldot Constraint')
    # plt.plot(np.delete(lam_exec,dont_show_id),np.delete(lamdot_act,dont_show_id), label='Recorded Joints Ldot Constraint',s=1)
    # plt.scatter(np.delete(lam_exec,dont_show_id),np.delete(lamdot_act,dont_show_id), label='Recorded Joints Ldot Constraint',s=1)
    plt.plot(lam_exec[1:],act_speed, label='Actual speed')
    # plt.plot(np.delete(lam_exec[1:],dont_show_id-1),np.delete(act_speed,dont_show_id-1), label='Actual speed')
    # plt.scatter(np.delete(lam_exec[1:],dont_show_id-1),np.delete(act_speed,dont_show_id-1), label='Actual speed',s=1,c='r')
    plt.title("CNT"+str(zone)+", Speed vs Lambda")
    plt.ylabel('Speed (mm/s)')
    plt.xlabel('Lambda (mm)')
    plt.legend()
    # plt.show()
    plt.savefig(data_dir+'speed'+"_cnt"+str(zone)+'.png')
    plt.clf()

    plt.plot(lam_exec,error_all)
    plt.title("CNT"+str(zone)+", Error vs Lambda")
    plt.ylabel('Projected Error (mm)')
    plt.xlabel('Lambda (mm)')
    # plt.show()
    plt.savefig(data_dir+'error'+"_cnt"+str(zone)+'.png')
    plt.clf()

    max_error_mat[mat_i]=np.max(error_all)
    ave_speed_mat[mat_i]=np.mean(act_speed)
    mat_i+=1


# plt.plot(lam_fit[0:-1:calc_ldot_step],lamdot_fit, label='Fitting')
# plt.title("Speed vs Lambda")
# plt.ylabel('Speed (mm/s)')
# plt.xlabel('Lambda (mm)')
# plt.legend()
# # plt.show()
# plt.savefig(data_dir+'lambdot_curve_js.png')
# plt.clf()

print(max_error_mat)
print(ave_speed_mat)

with open(data_dir+'max_error.npy','wb') as f:
    np.save(f,max_error_mat)
with open(data_dir+'ave_speed.npy','wb') as f:
    np.save(f,ave_speed_mat)