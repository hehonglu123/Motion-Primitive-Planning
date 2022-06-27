from pandas import read_csv, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import sys, copy
sys.path.append('../../../toolbox')
from robots_def import *
from utils import *
from toolbox_circular_fit import *
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

    return

    with open(data_dir+'error_'+case_file_name+'.npy','wb') as f:
        np.save(f,error)
    with open(data_dir+'normal_error_'+case_file_name+'.npy','wb') as f:
        np.save(f,angle_error)
    with open(data_dir+'speed_'+case_file_name+'.npy','wb') as f:
        np.save(f,act_speed)
    with open(data_dir+'lambda_'+case_file_name+'.npy','wb') as f:
        np.save(f,lam_exec)

robot=m900ia(d=50)
data_dir = 'data_opt_point_basis/'

client = FANUCClient()
utool_num = 2

with open(data_dir+'Curve_js.npy','rb') as f:
    curve_js = np.load(f)
with open(data_dir+'Curve_in_base_frame.npy','rb') as f:
    curve = np.load(f)
with open(data_dir+'Curve_R_in_base_frame.npy','rb') as f:
    R_all = np.load(f)
    curve_normal=R_all[:,:,-1]
curve = np.hstack((curve,curve_normal))

# motion parameters
# speed=100
# motion_type='movel'
motion_type='movel'
# speed=300
# all_speed=[300,1000]
# all_speed=[10,100]
# all_speed=[270]
all_speed=[200]
zone=100

# qp_cases=['qp','opt_init','opt','qp_heu','opt_init_heu','opt_heu']
# qp_cases=['nothing','init_10_200','opt_10_200']
# qp_cases=['qp_10_200_lowqddot']
# qp_cases=['qp_10_200_jerk','qp_10_250_jerk']
# qp_cases=['qp_10_200_jerk']
qp_cases=['nothing']

for speed in all_speed:
    for case in qp_cases:
        print(case)
        if case == 'nothing':
            total_seg = 10
            step = int((len(curve_js)-1)/total_seg)
            curve_js_plan = curve_js[::step]
        else:
            curve_js_plan = read_csv(data_dir+'curve_js_'+case+'.csv',header=None).values
            curve_js_plan=np.array(curve_js_plan).astype(float)

            # total_seg = 200
            # step = int((len(curve_js_plan)-1)/total_seg)
            # curve_js_plan = curve_js_plan[::step]

        # the original curve
        tp_pre = TPMotionProgram()
        j0 = jointtarget(1,1,utool_num,np.degrees(curve_js_plan[0]),[0]*6)
        tp_pre.moveJ(j0,50,'%',-1)
        j0 = jointtarget(1,1,utool_num,np.degrees(curve_js_plan[0]),[0]*6)
        tp_pre.moveJ(j0,5,'%',-1)
        client.execute_motion_program(tp_pre)

        tp = TPMotionProgram()
        for i in range(1,len(curve_js_plan)-1):
            this_zone = zone
            # if i==(len(curve_js_plan)-1)/2:
            #     this_zone = 40
            
            robt = jointtarget(1,1,utool_num,np.degrees(curve_js_plan[i]),[0]*6)
            if motion_type == 'movej':
                # tp.moveJ(robt,speed,'%',this_zone)
                tp.moveJ(robt,speed,'msec',this_zone)
            elif motion_type == 'movel':
                tp.moveL(robt,speed,'mmsec',this_zone)
                # tp.moveL(robt,speed,'msec',this_zone)

        robt_end = jointtarget(1,1,utool_num,np.degrees(curve_js_plan[-1]),[0]*6)
        if motion_type == 'movej':
            # tp.moveJ(robt_end,speed,'%',-1)
            tp.moveJ(robt_end,speed,'msec',-1)
        elif motion_type == 'movel':
            tp.moveL(robt_end,speed,'mmsec',-1)
            # tp.moveL(robt_end,speed,'msec',-1)
        # execute 
        res = client.execute_motion_program(tp)

        case_file_name = case+'_'+motion_type+'_'+str(speed)

        # Write log csv to file
        with open(data_dir+"curve_js_"+case_file_name+"_exe.csv","wb") as f:
            f.write(res)

        # read execution
        curve_js_exe = read_csv(data_dir+"curve_js_"+case_file_name+"_exe.csv",header=None).values[1:]
        curve_js_exe=np.array(curve_js_exe).astype(float)
        timestamp = curve_js_exe[:,0]*1e-3
        curve_js_exe = np.radians(curve_js_exe[:,1:])

        result_ana(curve_js_exe)