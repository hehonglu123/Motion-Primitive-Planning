import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
from robots_def import *
from lambda_calc import *
from error_check import *
import rpi_abb_irc5


class EGM_toolbox(object):
    ###need to tune delay on real robot
    def __init__(self,egm,robot=abb6640(d=50),delay=0.2):

        self.robot=robot
        self.egm=egm
        self.ts=0.004
        self.delay=delay

    def downsample24ms(self,curve,vd):
        lam=calc_lam_cs(curve[:,:3])

        steps=int((lam[-1]/vd)/self.ts)
        idx=np.linspace(0.,len(curve)-1,num=steps).astype(int)

        return idx

    def add_extension_egm(self,curve_cmd_js,extension_num=150):
        #################add extension#########################
        init_extension_js=np.linspace(curve_cmd_js[0]-extension_num*(curve_cmd_js[1]-curve_cmd_js[0]),curve_cmd_js[0],num=extension_num,endpoint=False)
        end_extension_js=np.linspace(curve_cmd_js[-1],curve_cmd_js[-1]+extension_num*(curve_cmd_js[-1]-curve_cmd_js[-2]),num=extension_num+1)[1:]

        init_extension_p=[]
        init_extension_R=[]
        for q in init_extension_js:
            pose_temp=self.robot.fwd(q)
            init_extension_p.append(pose_temp.p)
            init_extension_R.append(pose_temp.R)
        init_extension_p=np.array(init_extension_p)
        init_extension_R=np.array(init_extension_R)

        end_extension_p=[]
        end_extension_R=[]
        for q in end_extension_js:
            pose_temp=self.robot.fwd(q)
            end_extension_p.append(pose_temp.p)
            end_extension_R.append(pose_temp.R)
        end_extension_p=np.array(end_extension_p)
        init_extension_R=np.array(init_extension_R)

        curve_cmd_ext=np.vstack((init_extension_p,curve_cmd,end_extension_p))
        curve_cmd_R_ext=np.vstack((init_extension_R,curve_cmd_R,end_extension_R))

        curve_cmd_js_ext=np.vstack((init_extension_js,curve_cmd_js,end_extension_js))

        return curve_cmd_js_ext,curve_cmd_ext,curve_cmd_R_ext


    def add_extension_egm_cartesian(self,curve_cmd,curve_cmd_R,extension_num=150):
        #################add extension#########################
        init_extension_p=np.linspace(curve_cmd[0]-extension_num*(curve_cmd[1]-curve_cmd[0]),curve_cmd[0],num=extension_num,endpoint=False)
        end_extension_p=np.linspace(curve_cmd[-1],curve_cmd[-1]+extension_num*(curve_cmd[-1]-curve_cmd[-2]),num=extension_num+1)[1:]

        curve_cmd_ext=np.vstack((init_extension_p,curve_cmd,end_extension_p))
        curve_cmd_R_ext=np.vstack(([curve_cmd_R[0]]*extension_num,curve_cmd_R,[curve_cmd_R[-1]]*extension_num))

        return curve_cmd_ext,curve_cmd_R_ext


    def jog_joint_cartesian(self,p,R):
        self.clear_queue()
        res, state = self.egm.receive_from_robot(.1)
        q_cur=np.radians(state.joint_angles)
        pose_cur=self.robot.fwd(q_cur)
        num=2*int(np.linalg.norm(p-pose_cur.p)/(1000*self.ts))
        curve2start=np.linspace(pose_cur.p,p,num=num)
        R2start=orientation_interp(pose_cur.R,R,num)
        quat2start=[]
        for R in R2start:
            quat2start.append(R2q(R))

        try:
            for i in range(len(curve2start)):
                while True:
                    res_i, state_i = self.egm.receive_from_robot()

                    if res_i:
                        # print(state_i)
                        send_res = self.egm.send_to_robot_cart(curve2start[i], quat2start[i])
                        # print(send_res)
                        break

            for i in range(500):
                while True:
                    res_i, state_i = self.egm.receive_from_robot()
                    if res_i:
                        send_res = self.egm.send_to_robot_cart(p, R2q(R))
                        break

        except KeyboardInterrupt:
            raise

    def jog_joint(self,q):
        self.clear_queue()
        res, state = self.egm.receive_from_robot(.1)
        q_cur=np.radians(state.joint_angles)
        num=2*int(np.linalg.norm(q-q_cur)/(1000*self.ts))
        q2start=np.linspace(q_cur,q,num=num)

        
        try:
            for i in range(len(q2start)):
                while True:
                    res_i, state_i = self.egm.receive_from_robot()

                    if res_i:
                        # print(state_i)
                        send_res = self.egm.send_to_robot(q2start[i])
                        # print(send_res)
                        break

            for i in range(500):
                while True:
                    res_i, state_i = self.egm.receive_from_robot()
                    if res_i:
                        send_res = self.egm.send_to_robot(q)
                        break

        except KeyboardInterrupt:
            raise

    def traverse_curve_cartesian(self,curve,curve_R):
        curve_exe_js=[]
        timestamp=[]
        ###traverse curve
        print('traversing trajectory')
        try:
            for i in range(len(curve)):
                while True:
                    res_i, state_i = self.egm.receive_from_robot()
                    if res_i:
                        send_res = self.egm.send_to_robot_cart(curve[i], R2q(curve_R[i]))
                        #save joint angles
                        curve_exe_js.append(np.radians(state_i.joint_angles))
                        timestamp.append(state_i.robot_message.header.tm)
                        break
        except KeyboardInterrupt:
            raise
        
        timestamp=np.array(timestamp)/1000

        return timestamp,np.array(curve_exe_js)

    def clear_queue(self):
        # # Clear UDP queue
        while True:
            res_i, state_i = self.egm.receive_from_robot()
            if not res_i:
                break


def main():
    return
if __name__ == "__main__":
    main()