from pandas import read_csv, DataFrame
from general_robotics_toolbox import *
import sys, copy

sys.path.append('../../../toolbox')
sys.path.append('../../../circular_fit')
from robots_def import *
from utils import *
from toolbox_circular_fit import *
from robots_def import *
import matplotlib.pyplot as plt
from lambda_calc import *
from error_check import *
sys.path.append('../fanuc_toolbox')
from fanuc_client import *

# robot=abb6640(d=50)
robot=m900ia(d=50)
data_dir='data_qp_devel/'

client = FANUCClient()
utool_num = 2

# motion parameters
speed=100
zone=100

des_speed = 25 #mm/s

ang=30
##### create curve #####
###start rotation by 'ang' deg
start_p = np.array([2200,500, 1000])
end_p = np.array([2200, -500, 1000])
mid_p=(start_p+end_p)/2

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
        q_all=np.array(robot.inv(curve[i],R_all[i]))
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

# the original curve
tp_pre = TPMotionProgram()
j0 = jointtarget(1,1,utool_num,np.degrees(curve_js[0]),[0]*6)
tp_pre.moveJ(j0,50,'%',-1)
j0 = jointtarget(1,1,utool_num,np.degrees(curve_js[0]),[0]*6)
tp_pre.moveJ(j0,5,'%',-1)
client.execute_motion_program(tp_pre)

total_seg = 500
step=int((len(curve_js)-1)/total_seg)

tp = TPMotionProgram()
for i in range(step,len(curve_js)-1,step):
    robt = jointtarget(1,1,utool_num,np.degrees(curve_js[i]),[0]*6)
    # robt = joint2robtarget(curve_js[i],robot,1,1,utool_num)
    # for pose_i in range(3):
    #     robt.trans[pose_i]=curve[i][pose_i]
    #     robt.rot[pose_i]=R2wpr(R_all[i])[pose_i]

    # dt = round(np.linalg.norm(curve[i]-curve[i-step])/des_speed*1000)
    # tp.moveJ(robt,dt,'msec',zone)
    tp.moveJ(robt,speed,'%',zone)
    # tp.moveL(robt,200,'mmsec',zone)

    # if i>=24000 and i<26000:
    #     for j in range(50,step,50):
    #         robt = jointtarget(1,1,utool_num,np.degrees(curve_js[i+j]),[0]*6)
    #         # dt = round(np.linalg.norm(curve[i]-curve[i-step])/des_speed*1000)
    #         # tp.moveJ(robt,dt,'msec',zone)
    #         tp.moveJ(robt,speed,'%',zone)

robt = jointtarget(1,1,utool_num,np.degrees(curve_js[-1]),[0]*6)
# robt = joint2robtarget(curve_js[-1],robot,1,1,utool_num)
# dt = round(np.linalg.norm(curve[-1]-curve[-1-step])/des_speed*1000)
# tp.moveJ(robt,dt,'msec',-1)
tp.moveJ(robt,speed,'%',-1)
# tp.moveL(robt,200,'mmsec',-1)
# execute 
res = client.execute_motion_program(tp)

# Write log csv to file
with open(data_dir+"/curve_js_exe.csv","wb") as f:
    f.write(res)

# the executed in Joint space (FANUC m710ic)
col_names=['timestamp','J1', 'J2','J3', 'J4', 'J5', 'J6'] 
data=read_csv(data_dir+"/curve_js_exe.csv",names=col_names)
q1=data['J1'].tolist()[1:]
q2=data['J2'].tolist()[1:]
q3=data['J3'].tolist()[1:]
q4=data['J4'].tolist()[1:]
q5=data['J5'].tolist()[1:]
q6=data['J6'].tolist()[1:]
curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float))
timestamp=np.array(data['timestamp'].tolist()[1:]).astype(float)*1e-3 # from msec to sec

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
error,angle_error=calc_all_error_w_normal(curve_exe,curve,curve_exe_R[:,:,-1],curve_normal)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(lam_exec,act_speed, 'g-', label='Speed')
# ax1.plot(lam_exec[start_id:end_id],act_speed[start_id:end_id], 'r-', label='Speed')
ax2.plot(lam_exec, error, 'b-',label='Error')
ax2.plot(lam_exec, np.degrees(angle_error), 'y-',label='Normal Error')
ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Execution Result ")
ax1.legend(loc=6)
ax2.legend(loc=7)
plt.show()

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

plt.plot(curve[:,1],curve[:,0], 'red',label='Motion Cmd')
plt.plot(curve_exe[:,1],curve_exe[:,0], 'green',label='Executed Motion')
plt.axis('equal')
plt.gca().invert_xaxis()
plt.show()

lam=[0]
for i in range(1,len(curve)):
    lam.append(lam[-1]+np.linalg.norm(curve[i]-curve[i-1]))

lambdot_est = calc_lamdot(curve_js,lam,robot,step)
lambdot_act = calc_lamdot(curve_exe_js_act,lam_exec,robot,1)

plt.plot(lam[::step],lambdot_est,label='est ldot')
plt.plot(lam_exec,lambdot_act,label='act ldot')
plt.show()