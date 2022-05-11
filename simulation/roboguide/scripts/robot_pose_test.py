from math import radians,degrees
import numpy as np
from pandas import read_csv,DataFrame
from math import sin,cos

from general_robotics_toolbox import *
import sys
from matplotlib import pyplot as plt

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

def get_svJ(q_init,curve,curve_R,step,robot):
    curve = curve[::step]
    curve_R = curve_R[::step]
    all_svJ=[]
    
    q_last = q_init
    for i in range(1,len(curve)):
        q_all=np.array(robot.inv(curve[i],curve_R[i]))
        if len(q_all)==0:
            all_svJ=[0]
            break
        q_last = unwrapped_angle_check(q_last,q_all)
        J=robot.jacobian(q_last)
        all_svJ.append(np.min(np.linalg.svd(J)[1]))
    
    return all_svJ

# generate curve with linear orientation
# robot=abb6640(d=50)
robot=m900ia(d=50)
###generate a continuous arc, with linear orientation
start_p = np.array([2200,500, 1000])
end_p = np.array([2200, -500, 1000])
mid_p=(start_p+end_p)/2

# fanuc client
client = FANUCClient()
utool_num = 2

save_dir='../data/robot_pose_test/'

ang=30
zone=100
speed=1000

# all_ori = [0,90]
all_ori = [0,50]

# movel part
for curve_ori in all_ori:

    if curve_ori == 0:
        # all_pose_d_x = [0,-300,500]
        all_pose_d_x = [500]
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

        # translate and rotate the curve
        R=rox.rot([0,0,1],radians(curve_ori))
        new_curve = np.matmul(R,(curve-mid_p).T).T+mid_p+curve_pose
        curve=new_curve

        R_init=Ry(np.radians(135))
        R_end=Ry(np.radians(90))
        R_init=np.matmul(R,R_init)
        R_end=np.matmul(R,R_end)
        # exit()

        # interpolate orientation
        R_all=[R_init]
        k,theta=R2rot(np.dot(R_end,R_init.T))
        # theta=np.pi/4 #force 45deg change
        for i in range(1,len(curve)):
            angle=theta*i/(len(curve)-1)
            R_temp=rot(k,angle)
            R_all.append(np.dot(R_temp,R_init))

        #solve inv kin
        try:
            with open(save_dir+'Curve_js_movel_ori'+str(curve_ori)+'_x'+str(curve_pose[0])+'_y'+str(curve_pose[1])+'.npy','rb') as f:
                curve_js = np.load(f)
        except OSError as e:
            all_q_init = robot.inv(curve[0],R_init)
            argmin_q = -1
            max_min_svj = 0
            for i in range(len(all_q_init)):
                min_svj = np.min(get_svJ(all_q_init[i],curve,R_all,50,robot))
                if min_svj > max_min_svj:
                    max_min_svj=min_svj
                    argmin_q=i
            if argmin_q==-1:
                print("No solution")
                continue

            q_init=all_q_init[argmin_q]
            curve_js=[q_init]
            # theta=np.pi/4 #force 45deg change
            for i in range(1,len(curve)):
                q_all=np.array(robot.inv(curve[i],R_all[i]))
                ###choose inv_kin closest to previous joints
                curve_js.append(unwrapped_angle_check(curve_js[-1],q_all))

            curve_js=np.array(curve_js)
            R_all=np.array(R_all)
            # DataFrame(curve_js).to_csv(save_dir+'Curve_js_'+str(ang)+'.csv',header=False,index=False)
            with open(save_dir+'Curve_js_movel_ori'+str(curve_ori)+'_x'+str(curve_pose[0])+'_y'+str(curve_pose[1])+'.npy','wb') as f:
                np.save(f,curve_js)
        
        # tp program
        # move to start
        tp_pre = TPMotionProgram()
        j0 = jointtarget(1,1,utool_num,np.degrees(curve_js[0]),[0]*6)
        tp_pre.moveJ(j0,50,'%',-1)
        # robt_start = joint2robtarget(curve_js[0],robot,1,1,utool_num)
        robt_start = jointtarget(1,1,utool_num,np.degrees(curve_js[0]),[0]*6)
        tp_pre.moveL(robt_start,5,'mmsec',-1)
        client.execute_motion_program(tp_pre)
        # print(robt_start)
        # print(curve_js[0])

        tp = TPMotionProgram()
        # robt_mid = joint2robtarget(curve_js[int((len(curve)+1)/2)],robot,1,1,utool_num)
        robt_mid = jointtarget(1,1,utool_num,np.degrees(curve_js[int((len(curve)+1)/2)]),[0]*6)
        tp.moveL(robt_mid,speed,'mmsec',zone)
        # robt_end = joint2robtarget(curve_js[-1],robot,1,1,utool_num)
        robt_end = jointtarget(1,1,utool_num,np.degrees(curve_js[-1]),[0]*6)
        tp.moveL(robt_end,speed,'mmsec',-1)

        # print(robt_mid)
        # print(curve_js[int((len(curve)+1)/2)])
        # print(robt_end)
        # print(curve_js[-1])
        # execute 
        # print(curve_js[::500])
        res = client.execute_motion_program(tp)
        # Write log csv to file
        with open(save_dir+'Curve_exec_movel_ori'+str(curve_ori)+'_x'+str(curve_pose[0])+'_y'+str(curve_pose[1])+'.csv',"wb") as f:
            f.write(res)

exit()

# movec part
for curve_ori in all_ori:

    if curve_ori == 0:
        # all_pose_d_x = [0,-300,600]
        all_pose_d_x = [500]
        all_pose_d_y = [0,-1000,1000]
        all_pose = []
        for x in all_pose_d_x:
            for y in all_pose_d_y:
                p = np.array([x,y,0])
                all_pose.append(p)
    if curve_ori == 90:
        all_pose_d_x = [200]
        all_pose_d_y = [0,-1000,1000]
        all_pose = []
        for x in all_pose_d_x:
            for y in all_pose_d_y:
                p = np.array([x,y,0])
                all_pose.append(p)

    for curve_pose in all_pose:

        ##### create curve #####
        ###start rotation by 'ang' deg
        k=np.cross(end_p+np.array([0.1,0,0])-mid_p,start_p-mid_p)
        k=k/np.linalg.norm(k)
        theta=np.radians(ang)

        R=rot(k,theta)
        new_vec=R@(end_p-mid_p)
        new_end_p=mid_p+new_vec

        ###calculate lambda
        arc=arc_from_3point(start_p,end_p,mid_p,N=50001)
        arc1=arc[:int((len(arc)+1)/2)]
        arc2=arc[int((len(arc)+1)/2):]
        ###start rotation by 30 deg
        k=np.cross(end_p-arc2[0],start_p-arc2[0])
        k=k/np.linalg.norm(k)
        theta=np.radians(ang)
        R=rot(k,theta)

        arc2_new=np.zeros(arc2.shape)
        for i in range(len(arc2)):
            new_vec=R@(arc2[i]-arc2[0])
            arc2_new[i]=arc2[0]+new_vec

        curve=np.vstack((arc1,arc2_new))

        # translate and rotate the curve
        R=rox.rot([0,0,1],radians(curve_ori))
        new_curve = np.matmul(R,(curve-mid_p).T).T+mid_p+curve_pose
        curve=new_curve

        R_init=Ry(np.radians(135))
        R_end=Ry(np.radians(90))
        R_init=np.matmul(R,R_init)
        R_end=np.matmul(R,R_end)

        # interpolate orientation
        R_all=[R_init]
        k,theta=R2rot(np.dot(R_end,R_init.T))
        # theta=np.pi/4 #force 45deg change
        for i in range(1,len(curve)):
            angle=theta*i/(len(curve)-1)
            R_temp=rot(k,angle)
            R_all.append(np.dot(R_temp,R_init))

        #solve inv kin
        try:
            with open(save_dir+'Curve_js_movec_ori'+str(curve_ori)+'_x'+str(curve_pose[0])+'_y'+str(curve_pose[1])+'.npy','rb') as f:
                curve_js = np.load(f)
        except OSError as e:
            all_q_init = robot.inv(start_p,R_init)
            argmin_q = -1
            max_min_svj = 0
            for i in range(len(all_q_init)):
                min_svj = np.min(get_svJ(all_q_init[i],curve,R_all,50,robot))
                if min_svj > max_min_svj:
                    max_min_svj=min_svj
                    argmin_q=i
            if argmin_q==-1:
                print("No solution")
                continue
                
            q_init=all_q_init[argmin_q]
            curve_js=[q_init]
            # theta=np.pi/4 #force 45deg change
            for i in range(1,len(curve)):
                q_all=np.array(robot.inv(curve[i],R_all[i]))
                ###choose inv_kin closest to previous joints
                curve_js.append(unwrapped_angle_check(curve_js[-1],q_all))

            curve_js=np.array(curve_js)
            R_all=np.array(R_all)
            # DataFrame(curve_js).to_csv(save_dir+'Curve_js_'+str(ang)+'.csv',header=False,index=False)
            with open(save_dir+'Curve_js_movec_ori'+str(curve_ori)+'_x'+str(curve_pose[0])+'_y'+str(curve_pose[1])+'.npy','wb') as f:
                np.save(f,curve_js)
        
        # tp program
        # move to start
        tp_pre = TPMotionProgram()
        j0 = jointtarget(1,1,utool_num,np.degrees(curve_js[0]),[0]*6)
        tp_pre.moveJ(j0,50,'%',-1)
        robt_start = joint2robtarget(curve_js[0],robot,1,1,utool_num)
        tp_pre.moveL(robt_start,5,'mmsec',-1)
        client.execute_motion_program(tp_pre)

        tp = TPMotionProgram()
        robt_1 = joint2robtarget(curve_js[int(len(curve)/4)],robot,1,1,utool_num)
        robt_2 = joint2robtarget(curve_js[int((len(curve)+1)/2)],robot,1,1,utool_num)
        tp.moveC(robt_1,robt_2,speed,'mmsec',zone)
        robt_3 = joint2robtarget(curve_js[int(3*len(curve)/4)],robot,1,1,utool_num)
        robt_4 = joint2robtarget(curve_js[-1],robot,1,1,utool_num)
        tp.moveC(robt_3,robt_4,speed,'mmsec',-1)

        # execute 
        res = client.execute_motion_program(tp)
        # Write log csv to file
        with open(save_dir+'Curve_exec_movec_ori'+str(curve_ori)+'_x'+str(curve_pose[0])+'_y'+str(curve_pose[1])+'.npy',"wb") as f:
            f.write(res)