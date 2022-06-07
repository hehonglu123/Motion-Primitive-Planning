from pandas import read_csv, DataFrame
import sys, copy
sys.path.append('../../../toolbox')
sys.path.append('../../../circular_fit')
from robots_def import *
from utils import *
from toolbox_circular_fit import *
from abb_motion_program_exec_client import *
from robots_def import *
import matplotlib.pyplot as plt
from lambda_calc import *
from error_check import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

def result_ana(curve_exe_js):
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

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(lam_exec[1:],act_speed, 'g-', label='Speed')
    ax2.plot(lam_exec, error, 'b-',label='Error')
    ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')
    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
    plt.title("Execution Result (Speed/Error/Normal Error v.s. Lambda)")
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    plt.show()

robot=m900ia(d=50)
data_dir = 'data/'

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

# qp curve js
with open(data_dir+'Curve_js_qp2.npy','rb') as f:
    curve_js_qp = np.load(f)

curve_js_cmd = curve_js[::1000]

# motion parameters
speed=100
zone=100

# the original curve
tp_pre = TPMotionProgram()
j0 = jointtarget(1,1,utool_num,np.degrees(curve_js_cmd[0]),[0]*6)
tp_pre.moveJ(j0,50,'%',-1)
j0 = jointtarget(1,1,utool_num,np.degrees(curve_js_cmd[0]),[0]*6)
tp_pre.moveJ(j0,5,'%',-1)
client.execute_motion_program(tp_pre)

tp = TPMotionProgram()
for i in range(1,len(curve_js_cmd)-1):
    robt = jointtarget(1,1,utool_num,np.degrees(curve_js_cmd[i]),[0]*6)
    tp.moveJ(robt,speed,'%',zone)
robt_end = jointtarget(1,1,utool_num,np.degrees(curve_js_cmd[-1]),[0]*6)
tp.moveJ(robt_end,speed,'%',-1)
# execute 
res = client.execute_motion_program(tp)

# Write log csv to file
with open(data_dir+"/curve_js_exe.csv","wb") as f:
    f.write(res)

# read execution
curve_js_exe = read_csv(data_dir+"/curve_js_exe.csv",header=None).values[1:]
curve_js_exe=np.array(curve_js_exe).astype(float)
timestamp = curve_js_exe[:,0]*1e-3
curve_js_exe = np.radians(curve_js_exe[:,1:])

result_ana(curve_js_exe)

# the qp curve
tp_pre = TPMotionProgram()
j0 = jointtarget(1,1,utool_num,np.degrees(curve_js_qp[0]),[0]*6)
tp_pre.moveJ(j0,50,'%',-1)
j0 = jointtarget(1,1,utool_num,np.degrees(curve_js_qp[0]),[0]*6)
tp_pre.moveJ(j0,5,'%',-1)
client.execute_motion_program(tp_pre)

tp = TPMotionProgram()
for i in range(1,len(curve_js_qp)-1):
    robt = jointtarget(1,1,utool_num,np.degrees(curve_js_qp[i]),[0]*6)
    tp.moveJ(robt,speed,'%',zone)
robt_end = jointtarget(1,1,utool_num,np.degrees(curve_js_qp[-1]),[0]*6)
tp.moveJ(robt_end,speed,'%',-1)
# execute 
res = client.execute_motion_program(tp)

# Write log csv to file
with open(data_dir+"/curve_js_qp_exe.csv","wb") as f:
    f.write(res)

# read execution
curve_js_exe = read_csv(data_dir+"/curve_js_qp_exe.csv",header=None).values[1:]
curve_js_exe=np.array(curve_js_exe).astype(float)
timestamp = curve_js_exe[:,0]*1e-3
curve_js_exe = np.radians(curve_js_exe[:,1:])

result_ana(curve_js_exe)

