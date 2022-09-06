import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from io import StringIO
from threading import Thread

from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *

dataset='wood/'
solution_dir='diffevo3/'
data_dir="../data/"+dataset
cmd_dir=data_dir+'dual_arm/'+solution_dir+'50L/'
relative_path = read_csv(data_dir+"/Curve_dense.csv", header=None).values
lam_relative=calc_lam_cs(relative_path)

with open(data_dir+'dual_arm/'+solution_dir+'abb1200.yaml') as file:
    H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

base2_R=H_1200[:3,:3]
base2_p=1000*H_1200[:-1,-1]

with open(data_dir+'dual_arm/'+solution_dir+'tcp.yaml') as file:
    H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
robot1=abb6640(d=50)
robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])
ms1 = MotionSend(robot1=robot1,robot2=robot2,base2_R=base2_R,base2_p=base2_p,url='http://192.168.55.1:80')
ms2=MotionSend(robot1=robot2, url='http://192.168.55.3:80')


df1=None
df2=None
def move_robot1():
    global df1, ms1, robot1,primitives1,breakpoints1,p_bp1,q_bp1,v1_all,z1_all
    print(v1_all)
    logged_data= ms1.exec_motions(robot1,primitives1,breakpoints1,p_bp1,q_bp1,v1_all,z1_all)
    StringData=StringIO(logged_data)
    df1 = read_csv(StringData, sep =",")

def move_robot2():
    global df2, ms2, robot2,primitives2,breakpoints2,p_bp2,q_bp2,v2_all,z2_all
    logged_data= ms2.exec_motions(robot2,primitives2,breakpoints2,p_bp2,q_bp2,v2_all,z2_all)
    StringData=StringIO(logged_data)
    df2 = read_csv(StringData, sep =",")


def main():
    global ms1,ms2,df1, df2, robot1,primitives1,breakpoints1,p_bp1,q_bp1,v1_all,z1_all, robot2,primitives2,breakpoints2,p_bp2,q_bp2,v2_all,z2_all

    

    ###extract data commands
    breakpoints1,primitives1,p_bp1,q_bp1=ms1.extract_data_from_cmd(cmd_dir+'command1.csv')
    breakpoints2,primitives2,p_bp2,q_bp2=ms2.extract_data_from_cmd(cmd_dir+'command2.csv')
    breakpoints1[1:]=breakpoints1[1:]-1
    breakpoints2[2:]=breakpoints2[2:]-1
    ###get dt per segment
    v_relative=800
    dt=(lam_relative[breakpoints1[5]]-lam_relative[breakpoints1[4]])/v_relative
    print('dt',dt)
    ###extension
    p_bp1,q_bp1,p_bp2,q_bp2=ms1.extend_dual(robot1,p_bp1,q_bp1,primitives1,robot2,p_bp2,q_bp2,primitives2,breakpoints1)

    ###calculate desired TCP speed for both arm
    curve_js1=read_csv(data_dir+'/dual_arm/'+solution_dir+'arm1.csv', header=None).values
    curve_js2=read_csv(data_dir+'/dual_arm/'+solution_dir+'arm2.csv', header=None).values

    lam1=calc_lam_js(curve_js1,robot1)
    lam2=calc_lam_js(curve_js2,robot2)

    speed_ratio=[]      ###speed of robot1 TCP / robot2 TCP
    for i in range(1,len(breakpoints1)):
        speed_ratio.append((lam1[breakpoints1[i]-1]-lam1[breakpoints1[i-1]-1])/(lam2[breakpoints1[i]-1]-lam2[breakpoints1[i-1]-1]))

    ###specify speed here for robot2
    s2_all=[200]
    v2_all=[v200]
    z1_all=[z10]*len(breakpoints1)
    z2_all=[z10]*len(breakpoints2)

    v1_all=[v200]
    s1_all=[200]
    for v_idx in range(len(s2_all)-1):

        s2_all.append((lam2[breakpoints2[v_idx+1]]-lam2[breakpoints2[v_idx]])/dt)
        v2_all.append(speeddata(s2_all[v_idx+1],9999999,9999999,999999))
        s1=s2_all[v_idx+1]*speed_ratio[v_idx]
        s1_all.append(s1)
        v1 = speeddata(s1,9999999,9999999,999999)
        v1_all.append(v1)


    ###move to start first 

    ms1.exec_motions(robot1,[primitives1[0]],[breakpoints1[0]],[p_bp1[0]],[q_bp1[0]],vmax,fine)
    ms2.exec_motions(robot2,[primitives2[0]],[breakpoints2[0]],[p_bp2[0]],[q_bp2[0]],vmax,fine)

    t1 = Thread(target=move_robot1)
    t2 = Thread(target=move_robot2)

    # start the threads
    t1.start()
    t2.start()

    # wait for the threads to complete
    t1.join()
    t2.join()

    lam_exe1, curve_exe1, curve_exe_R1,curve_exe_js1, speed1, timestamp1=ms1.logged_data_analysis(robot1,df1,realrobot=True)
    lam_exe2, curve_exe2, curve_exe_R2,curve_exe_js2, speed2, timestamp2=ms2.logged_data_analysis(robot2,df2,realrobot=True)

    if len(curve_exe_js1)>len(curve_exe_js2):
        print('size mismatch, padding now')
        # speed2=np.append(speed2,[0]*(len(curve_exe_js1)-len(curve_exe_js2)))
        curve_exe_R1=np.append(curve_exe_R1,[curve_exe_R1[-1]]*(len(curve_exe_js1)-len(curve_exe_js2)))
        curve_exe_js2=np.pad(curve_exe_js2,[(0,len(curve_exe_js1)-len(curve_exe_js2)),(0,0)], mode="edge")
        curve_exe2=np.pad(curve_exe2,[(0,len(curve_exe1)-len(curve_exe2)),(0,0)], mode="edge")
        
    elif len(curve_exe_js1)<len(curve_exe_js2):
        print('size mismatch, padding now')
        # speed1=np.append(speed1,[0]*(len(curve_exe_js2)-len(curve_exe_js1)))
        curve_exe_R2=np.append(curve_exe_R2,[curve_exe_R2[-1]]*(len(curve_exe_js2)-len(curve_exe_js1)))
        curve_exe_js1=np.pad(curve_exe_js1,[(0,len(curve_exe_js2)-len(curve_exe_js1)),(0,0)], mode="edge")
        curve_exe1=np.pad(curve_exe1,[(0,len(curve_exe2)-len(curve_exe1)),(0,0)], mode="edge")

    relative_path_exe,relative_path_exe_R=form_relative_path(robot1,robot2,curve_exe_js1,curve_exe_js2,base2_R,base2_p)
    lam=calc_lam_cs(relative_path_exe)
    speed=np.gradient(lam)/np.gradient(timestamp1)

    lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe, relative_path_exe_R=\
            ms1.chop_extension_dual(lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp1, relative_path_exe,relative_path_exe_R,relative_path[0,:3],relative_path[-1,:3])

    speed1=get_speed(curve_exe1,timestamp)
    speed2=get_speed(curve_exe2,timestamp)

    error,angle_error=calc_all_error_w_normal(relative_path_exe,relative_path[:,:3],relative_path_exe_R[:,:,-1],relative_path[:,3:])


    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(lam,speed, 'g-', label='Relative Speed')
    ax1.plot(lam,speed1, 'r-', label='TCP1 Speed')
    ax1.plot(lam,speed2, 'm-', label='TCP2 Speed')
    ax2.plot(lam, error, 'b-',label='Error')
    ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

    ax1.set_xlabel('lambda (mm)')
    ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
    ax2.set_ylabel('Error (mm)', color='b')
    plt.title('Uncoordinated')
    ax1.legend(loc=0)

    ax2.legend(loc=0)

    plt.legend()
    # plt.savefig('recorded_data/curve_exe_v'+str(s2)+'_z'+str(zone))
    plt.show()
if __name__ == "__main__":
    main()