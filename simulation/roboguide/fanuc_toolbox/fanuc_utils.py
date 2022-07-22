import numpy as np
from pandas import read_csv, DataFrame
from fanuc_motion_program_exec_client import *
from general_robotics_toolbox import *

sys.path.append('../../../toolbox/')
from utils import *
from lambda_calc import *
from robots_def import *

class MotionSendFANUC(object):
    def __init__(self,group=1,uframe=1,utool=2,robot_ip='127.0.0.2',robot1=m710ic(d=50),robot2=m900ia(),group2=2,uframe2=1,utool2=2) -> None:
        
        self.client = FANUCClient(robot_ip)
        
        # robot1 roboguide info
        self.group = group
        self.uframe = uframe
        self.utool = utool

        # robot2 roboguide info
        self.group2 = group2
        self.uframe2 = uframe2
        self.utool2 = utool2

        # robots
        self.robot1=robot1
        self.robot2=robot2
    
    def exec_motions(self,robot,primitives,breakpoints,p_bp,q_bp,speed,zone):

        tp_pre = TPMotionProgram()

        # move to start
        j0 = joint2robtarget(q_bp[0][0],robot,self.group,self.uframe,self.utool)
        tp_pre.moveJ(j0,50,'%',-1)
        tp_pre.moveJ(j0,5,'%',-1)
        self.client.execute_motion_program(tp_pre)

        # start traj
        tp = TPMotionProgram()
        for i in range(1,len(primitives)):
            if i == len(primitives)-1:
                this_zone = -1
            else:
                this_zone = zone
            if primitives[i]=='movel_fit':
                robt = joint2robtarget(q_bp[i][0],robot,self.group,self.uframe,self.utool)
                tp.moveL(robt,speed,'mmsec',this_zone)
            elif primitives[i]=='movec_fit':
                robt_mid = joint2robtarget(q_bp[i][0],robot,self.group,self.uframe,self.utool)
                robt = joint2robtarget(q_bp[i][1],robot,self.group,self.uframe,self.utool)
                tp.moveC(robt_mid,robt,speed,'mmsec',this_zone)
            else: #moveJ
                robt = jointtarget(self.group,self.uframe,self.utool,np.degrees(q_bp[i][0]),[0]*6)
                tp.moveJ(robt,speed,'%',this_zone)
        
        return self.client.execute_motion_program(tp)

    def exec_motions_multimove(self,robot1,robot2,primitives1,primitives2,p_bp1,p_bp2,q_bp1,q_bp2,speed,zone):

        tp_follow = TPMotionProgram()
        tp_lead = TPMotionProgram()
        #### move to start
        # robot1
        j0 = joint2robtarget(q_bp1[0][0],robot1,self.group,self.uframe,self.utool)
        tp_follow.moveJ(j0,50,'%',-1)
        tp_follow.moveJ(j0,5,'%',-1)
        # robot2
        j0 = joint2robtarget(q_bp2[0][0],robot2,self.group2,self.uframe2,self.utool2)
        tp_lead.moveJ(j0,50,'%',-1)
        tp_lead.moveJ(j0,5,'%',-1)
        self.client.execute_motion_program_coord(tp_lead,tp_follow)

        #### start traj
        tp_follow = TPMotionProgram()
        tp_lead = TPMotionProgram()
        for i in range(1,len(primitives1)):
            if i == len(primitives1)-1:
                this_zone = -1
            else:
                this_zone = zone
            if primitives1[i]=='movel_fit':
                # robot1
                robt1 = joint2robtarget(q_bp1[i][0],robot1,self.group,self.uframe,self.utool)
                tp_follow.moveL(robt1,speed,'mmsec',this_zone,'COORD')
                # robot2
                robt2 = joint2robtarget(q_bp2[i][0],robot2,self.group2,self.uframe2,self.utool2)
                tp_lead.moveL(robt2,speed,'mmsec',this_zone,'COORD')
            elif primitives1[i]=='movec_fit':
                # robot1
                robt_mid1 = joint2robtarget(q_bp1[i][0],robot1,self.group,self.uframe,self.utool)
                robt1 = joint2robtarget(q_bp1[i][1],robot1,self.group,self.uframe,self.utool)
                tp_follow.moveC(robt_mid1,robt1,speed,'mmsec',this_zone,'COORD')
                # robot2
                robt_mid2 = joint2robtarget(q_bp2[i][0],robot2,self.group2,self.uframe2,self.utool2)
                robt2 = joint2robtarget(q_bp2[i][1],robot2,self.group2,self.uframe2,self.utool2)
                tp_lead.moveC(robt_mid2,robt2,speed,'mmsec',this_zone,'COORD')
            # else: #moveJ
            #     robt = jointtarget(self.group,self.uframe,self.utool,np.degrees(q_bp[i][0]),[0]*6)
            #     tp.moveJ(robt,speed,'%',this_zone)
        
        return self.client.execute_motion_program_coord(tp_lead,tp_follow)
    
    def logged_data_analysis(self,robot,df):

        q1=df['J1'].tolist()[1:]
        q2=df['J2'].tolist()[1:]
        q3=df['J3'].tolist()[1:]
        q4=df['J4'].tolist()[1:]
        q5=df['J5'].tolist()[1:]
        q6=df['J6'].tolist()[1:]
        curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float))
        timestamp=np.array(df['timestamp'].tolist()[1:]).astype(float)*1e-3 # from msec to sec

        act_speed=[0]
        lam_exec=[0]
        curve_exe=[]
        curve_exe_R=[]
        curve_exe_js_act=[]
        timestamp_act = []
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
            timestamp_act.append(timestamp[i])
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
        act_speed = np.array(act_speed)

        return lam_exec, curve_exe, curve_exe_R,curve_exe_js_act, act_speed, timestamp_act
    
    def logged_data_analysis_multimove(self,df,base2_R,base2_p,realrobot=False):
        q1_1=df['J11'].tolist()[1:-1]
        q1_2=df['J12'].tolist()[1:-1]
        q1_3=df['J13'].tolist()[1:-1]
        q1_4=df['J14'].tolist()[1:-1]
        q1_5=df['J15'].tolist()[1:-1]
        q1_6=df['J16'].tolist()[1:-1]
        q2_1=df['J21'].tolist()[1:-1]
        q2_2=df['J22'].tolist()[1:-1]
        q2_3=df['J23'].tolist()[1:-1]
        q2_4=df['J24'].tolist()[1:-1]
        q2_5=df['J25'].tolist()[1:-1]
        q2_6=df['J26'].tolist()[1:-1]
        timestamp=np.array(df['timestamp'].tolist()[1:]).astype(float)*1e-3 # from msec to sec

        curve_exe_js1=np.radians(np.vstack((q1_1,q1_2,q1_3,q1_4,q1_5,q1_6)).T.astype(float))
        curve_exe_js2=np.radians(np.vstack((q2_1,q2_2,q2_3,q2_4,q2_5,q2_6)).T.astype(float))

        if realrobot:
            timestamp, curve_exe_js_all=lfilter(timestamp, np.hstack((curve_exe_js1,curve_exe_js2)))
            curve_exe_js1=curve_exe_js_all[:,:6]
            curve_exe_js2=curve_exe_js_all[:,6:]

        act_speed=[0]
        lam=[0]
        relative_path_exe=[]
        relative_path_exe_R=[]
        curve_exe1=[]
        curve_exe2=[]
        curve_exe_R1=[]
        curve_exe_R2=[]
        curve_exe_js1_act=[]
        curve_exe_js2_act=[]
        timestamp_act = []
        last_cont = False
        for i in range(len(curve_exe_js1)):
            if i>5 and i<len(curve_exe_js1)-5:
                # if the recording is not fast enough
                # then having to same logged joint angle
                # do interpolation for estimation
                if np.all(curve_exe_js1[i]==curve_exe_js1[i+1]) and np.all(curve_exe_js2[i]==curve_exe_js2[i+1]):
                    last_cont = True
                    continue

            curve_exe_js1_act.append(curve_exe_js1[i])
            curve_exe_js2_act.append(curve_exe_js2[i])
            timestamp_act.append(timestamp[i])
            pose1_now=self.robot1.fwd(curve_exe_js1[i])
            pose2_now=self.robot2.fwd(curve_exe_js2[i])
            # curve in robot's own frame
            curve_exe1.append(pose1_now.p)
            curve_exe2.append(pose2_now.p)
            curve_exe_R1.append(pose1_now.R)
            curve_exe_R2.append(pose2_now.R)

            pose2_world_now=self.robot2.fwd(curve_exe_js2[i],base2_R,base2_p)
            relative_path_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))
            relative_path_exe_R.append(pose2_world_now.R.T@pose1_now.R)
            
            if i>0:
                lam.append(lam[-1]+np.linalg.norm(relative_path_exe[-1]-relative_path_exe[-2]))
            try:
                if timestamp[-1]!=timestamp[-2]:
                    if last_cont:
                        timestep=timestamp[i]-timestamp[i-2]
                    else:
                        timestep=timestamp[i]-timestamp[i-1]
                    act_speed.append(np.linalg.norm(relative_path_exe[-1]-relative_path_exe[-2])/timestep)
                    
            except IndexError:
                pass
            last_cont = False

        return np.array(lam), np.array(curve_exe1),np.array(curve_exe2), np.array(curve_exe_R1),np.array(curve_exe_R2),curve_exe_js1_act,curve_exe_js2_act, act_speed, timestamp_act, np.array(relative_path_exe), np.array(relative_path_exe_R)

    def chop_extension(self,curve_exe, curve_exe_R,curve_exe_js, speed, timestamp,curve,curve_normal):
        start_idx=np.argmin(np.linalg.norm(curve[0,:]-curve_exe,axis=1))+1
        end_idx=np.argmin(np.linalg.norm(curve[-1,:]-curve_exe,axis=1))

        #make sure extension doesn't introduce error
        if np.linalg.norm(curve_exe[start_idx]-curve[0,:])>0.05:
            start_idx+=1
        if np.linalg.norm(curve_exe[end_idx]-curve[-1,:])>0.05:
            end_idx-=1

        curve_exe=curve_exe[start_idx:end_idx+1]
        curve_exe_js=curve_exe_js[start_idx:end_idx+1]
        curve_exe_R=curve_exe_R[start_idx:end_idx+1]
        speed=speed[start_idx:end_idx+1]
        timestamp=timestamp[start_idx:end_idx+1]
        lam=calc_lam_cs(curve_exe)
        return lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp
    
    def chop_extension_dual(self,lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R,p_start,p_end):
        start_idx=np.argmin(np.linalg.norm(p_start-relative_path_exe,axis=1))+1
        end_idx=np.argmin(np.linalg.norm(p_end-relative_path_exe,axis=1))

        #make sure extension doesn't introduce error
        if np.linalg.norm(relative_path_exe[start_idx]-p_start)>0.5:
            start_idx+=1
        if np.linalg.norm(relative_path_exe[end_idx]-p_end)>0.5:
            end_idx-=1

        curve_exe1=curve_exe1[start_idx:end_idx+1]
        curve_exe2=curve_exe2[start_idx:end_idx+1]
        curve_exe_R1=curve_exe_R1[start_idx:end_idx+1]
        curve_exe_R2=curve_exe_R2[start_idx:end_idx+1]
        curve_exe_js1=curve_exe_js1[start_idx:end_idx+1]
        curve_exe_js2=curve_exe_js2[start_idx:end_idx+1]

        relative_path_exe=relative_path_exe[start_idx:end_idx+1]
        relative_path_exe_R=relative_path_exe_R[start_idx:end_idx+1]

        speed=speed[start_idx:end_idx+1]
        lam=calc_lam_cs(relative_path_exe)

        return lam, curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,curve_exe_js1,curve_exe_js2, speed, timestamp, relative_path_exe,relative_path_exe_R
    
    def form_relative_path(self,curve_js1,curve_js2,base2_R,base2_p):
        relative_path_exe=[]
        relative_path_exe_R=[]
        curve_exe1=[]
        curve_exe2=[]
        curve_exe_R1=[]
        curve_exe_R2=[]
        for i in range(len(curve_js1)):
            pose1_now=self.robot1.fwd(curve_js1[i])
            pose2_now=self.robot2.fwd(curve_js2[i])

            curve_exe1.append(pose1_now.p)
            curve_exe2.append(pose2_now.p)
            curve_exe_R1.append(pose1_now.R)
            curve_exe_R2.append(pose2_now.R)

            pose2_world_now=self.robot2.fwd(curve_js2[i],base2_R,base2_p)


            relative_path_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))
            relative_path_exe_R.append(pose2_world_now.R.T@pose1_now.R)
        return np.array(relative_path_exe),np.array(relative_path_exe_R)

    def calc_robot2_q_from_blade_pose(self,blade_pose,base2_R,base2_p):
        R2=base2_R.T@blade_pose[:3,:3]
        p2=-base2_R.T@(base2_p-blade_pose[:3,-1])

        return self.robot2.inv(p2,R2)[1]
    
    def write_data_to_cmd(self,filename,breakpoints,primitives, p_bp,q_bp):
        p_bp_new=[]
        q_bp_new=[]
        for i in range(len(breakpoints)):
            if len(p_bp[i])==2:
                p_bp_new.append([np.array(p_bp[i][0]),np.array(p_bp[i][1])])
                q_bp_new.append([np.array(q_bp[i][0]),np.array(q_bp[i][1])])
            else:
                p_bp_new.append([np.array(p_bp[i][0])])
                q_bp_new.append([np.array(q_bp[i][0])])
        df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'points':p_bp_new,'q_bp':q_bp_new})
        df.to_csv(filename,header=True,index=False)

    def extend_start_end(self,robot,q_bp,primitives,breakpoints,points_list,extension_start=100,extension_end=100):
        
        dist_bp = np.linalg.norm(np.array(points_list[0][0])-np.array(points_list[1][0]))
        if dist_bp > extension_start:
            dist_bp=extension_start
        step_to_extend=round(extension_start/dist_bp)
        extend_step_d_start=float(extension_start)/step_to_extend
        
        ###initial point extension
        pose_start=robot.fwd(q_bp[0][-1])
        p_start=pose_start.p
        R_start=pose_start.R
        pose_end=robot.fwd(q_bp[1][-1])
        p_end=pose_end.p
        R_end=pose_end.R
        if primitives[1]=='movel_fit':
            #find new start point
            slope_p=p_end-p_start
            slope_p=slope_p/np.linalg.norm(slope_p)
            p_start_new=p_start-extension_start*slope_p        ###extend 5cm backward

            #find new start orientation
            k,theta=R2rot(R_end@R_start.T)
            theta_new=-extension_start*theta/np.linalg.norm(p_end-p_start)
            # R_start_new=rot(k,theta_new)@R_start

            # adding extension with uniform space
            for i in range(1,step_to_extend+1):
                p_extend=p_start-i*extend_step_d_start*slope_p
                theta_extend=-np.linalg.norm(p_extend-p_start)*theta/np.linalg.norm(p_end-p_start)
                R_extend=rot(k,theta_extend)@R_start
                points_list.insert(0,[p_extend])
                q_bp.insert(0,[car2js(robot,q_bp[0][0],p_extend,R_extend)[0]])
                primitives.insert(1,'movel_fit')

            #solve invkin for initial point
            # points_list[0][0]=p_start_new
            # q_bp[0][0]=car2js(robot,q_bp[0][0],p_start_new,R_start_new)[0]

        elif  primitives[1]=='movec_fit':
            #define circle first
            pose_mid=robot.fwd(q_bp[0][0])
            p_mid=pose_mid.p
            R_mid=pose_mid.R
            center, radius=circle_from_3point(p_start,p_end,p_mid)

            #find desired rotation angle
            angle=extension_start/radius

            #find new start point
            plane_N=np.cross(p_end-center,p_start-center)
            plane_N=plane_N/np.linalg.norm(plane_N)
            R_temp=rot(plane_N,angle)
            p_start_new=center+R_temp@(p_start-center)

            #find new start orientation
            k,theta=R2rot(R_end@R_start.T)
            theta_new=-extension_start*theta/np.linalg.norm(p_end-p_start)
            R_start_new=rot(k,theta_new)@R_start

            #solve invkin for initial point
            points_list[0][0]=p_start_new
            q_bp[0][0]=car2js(robot,q_bp[0][0],p_start_new,R_start_new)[0]

        else:
            #find new start point
            J_start=robot.jacobian(q_bp[0][0])
            qdot=q_bp[1][0]-q_bp[0][0]
            v=J_start[3:,:]@qdot
            t=extension_start/np.linalg.norm(v)
            
            q_bp[0][0]=q_bp[0][0]+qdot*t
            points_list[0][0]=robot.fwd(q_bp[0][0]).p

        ###end point extension
        pose_start=robot.fwd(q_bp[-2][-1])
        p_start=pose_start.p
        R_start=pose_start.R
        pose_end=robot.fwd(q_bp[-1][-1])
        p_end=pose_end.p
        R_end=pose_end.R
        extend_step_d_end=float(extension_end)/step_to_extend

        if primitives[-1]=='movel_fit':
            #find new end point
            slope_p=p_end-p_start
            slope_p=slope_p/np.linalg.norm(slope_p)
            # p_end_new=p_end+extension_d*slope_p        ###extend 5cm backward

            #find new end orientation
            k,theta=R2rot(R_end@R_start.T)
            # theta_new=extension_d*theta/np.linalg.norm(p_end-p_start)
            # R_end_new=rot(k,theta_new)@R_end

            # adding extension with uniform space
            for i in range(1,step_to_extend+1):
                p_extend=p_end+i*extend_step_d_end*slope_p
                theta_extend=np.linalg.norm(p_extend-p_end)*theta/np.linalg.norm(p_end-p_start)
                R_extend=rot(k,theta_extend)@R_end
                points_list.append([p_extend])
                q_bp.append([car2js(robot,q_bp[0][0],p_extend,R_extend)[0]])
                primitives.append('movel_fit')

            #solve invkin for end point
            # q_bp[-1][-1]=car2js(robot,q_bp[-1][0],p_end_new,R_end_new)[0]
            # points_list[-1][0]=p_end_new


        elif  primitives[1]=='movec_fit':
            #define circle first
            pose_mid=robot.fwd(q_bp[-1][0])
            p_mid=pose_mid.p
            R_mid=pose_mid.R
            center, radius=circle_from_3point(p_start,p_end,p_mid)

            #find desired rotation angle
            angle=extension_end/radius

            #find new end point
            plane_N=np.cross(p_start-center,p_end-center)
            plane_N=plane_N/np.linalg.norm(plane_N)
            R_temp=rot(plane_N,angle)
            p_end_new=center+R_temp@(p_end-center)

            #find new end orientation
            k,theta=R2rot(R_end@R_start.T)
            theta_new=extension_end*theta/np.linalg.norm(p_end-p_start)
            R_end_new=rot(k,theta_new)@R_end

            #solve invkin for end point
            q_bp[-1][-1]=car2js(robot,q_bp[-1][-1],p_end_new,R_end_new)[0]
            points_list[-1][-1]=p_end_new   #midpoint not changed

        else:
            #find new end point
            J_end=robot.jacobian(q_bp[-1][0])
            qdot=q_bp[-1][0]-q_bp[-2][0]
            v=J_end[3:,:]@qdot
            t=extension_end/np.linalg.norm(v)
            
            q_bp[-1][-1]=q_bp[-1][-1]+qdot*t
            points_list[-1][0]=robot.fwd(q_bp[-1][-1]).p

        return primitives,points_list,q_bp

    def extend_dual(self,robot1,p_bp1,q_bp1,primitives1,robot2,p_bp2,q_bp2,primitives2,breakpoints,extension_d=100):
        #extend porpotionally
        d1_start=np.linalg.norm(p_bp1[1][-1]-p_bp1[0][-1])
        d2_start=np.linalg.norm(p_bp2[1][-1]-p_bp2[0][-1])
        d1_end=np.linalg.norm(p_bp1[-1][-1]-p_bp1[-2][-1])
        d2_end=np.linalg.norm(p_bp2[-1][-1]-p_bp2[-2][-1])

        primitives1,p_bp1,q_bp1=self.extend_start_end(robot1,q_bp1,primitives1,breakpoints,p_bp1,extension_start=extension_d*d1_start/d2_start,extension_end=extension_d*d1_end/d2_end)
        primitives2,p_bp2,q_bp2=self.extend_start_end(robot2,q_bp2,primitives2,breakpoints,p_bp2,extension_start=extension_d,extension_end=extension_d)

        return p_bp1,q_bp1,p_bp2,q_bp2

    def extract_data_from_cmd(self,filename):
        data = read_csv(filename)

        if 'breakpoints' in data.keys():
            breakpoints=np.array(data['breakpoints'].tolist())
        else:
            breakpoints=[]
        primitives=data['primitives'].tolist()
        points=data['points'].tolist()
        qs=data['q_bp'].tolist()
        p_bp=[]
        q_bp=[]
        for i in range(len(primitives)):
            if primitives[i]=='movel_fit':
                point=self.extract_points(primitives[i],points[i])
                p_bp.append([point])
                q=self.extract_points(primitives[i],qs[i])
                q_bp.append([q])


            elif primitives[i]=='movec_fit':
                point1,point2=self.extract_points(primitives[i],points[i])
                p_bp.append([point1,point2])
                q1,q2=self.extract_points(primitives[i],qs[i])
                q_bp.append([q1,q2])

            else:
                point=self.extract_points(primitives[i],points[i])
                p_bp.append([point])
                q=self.extract_points(primitives[i],qs[i])
                q_bp.append([q])

        return breakpoints,primitives, p_bp,q_bp

    def extract_points(self,primitive_type,points):
        if primitive_type=='movec_fit':
            endpoints=points[8:-3].split('array')
            endpoint1=endpoints[0][:-4].split(',')
            endpoint2=endpoints[1][2:].split(',')

            return list(map(float, endpoint1)),list(map(float, endpoint2))
        else:
            if points[1] == '[':
                endpoint=points[2:-2].split(',')
                return np.array(list(map(float, endpoint)))
            else:
                endpoint=points[8:-3].split(',')
                return np.array(list(map(float, endpoint)))