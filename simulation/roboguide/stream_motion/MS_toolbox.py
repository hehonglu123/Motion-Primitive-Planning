from copy import deepcopy
import time
import numpy as np
from numpy import deg2rad,rad2deg
from urllib.request import urlopen  
from fanuc_stream import *

from pandas import *
from matplotlib import pyplot as plt
import sys
sys.path.append('../../../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *

class MS_toolbox(object):
    def __init__(self,ms,robot) -> None:
        
        self.robot=robot
        self.ms=ms
        self.dt=0.008
    
    def start_ms(self):

        # DO[102]=ON
        motion_url='http://'+self.ms.robot_ip+'/kcl/set%20port%20dout%20[102]%20=%20on'
        res = urlopen(motion_url)

        # send an end packet to reset
        self.ms.end_stream()
        # clear queue
        self.ms.clear_queue()
        # send start packet
        self.ms.start_stream()

        # wait for robot to be ready
        try:
            while True:
                while True:
                    res,status = self.ms.receive_from_robot()
                    if res:
                        break
                if self.ms.robot_ready:
                    break
        except KeyboardInterrupt:
            print("Interuptted")
            self.ms.end_stream()
            exit()
    
    def _send_curve(self,curve_js):

        curve_js_exe=[]
        curve_ts = []
        # send cmd
        self.ms.clear_queue()
        curve_js_exe=[]
        try:
            for i in range(len(curve_js)):
                while True:
                    res,status = self.ms.receive_from_robot()
                    if res:
                        send_res = self.ms.send_joint(curve_js[i])
                        curve_js_exe.append(status[2])
                        curve_ts.append(status[1])
                        break
            for i in range(10): # 50 ms delay
                while True:
                    res,status = self.ms.receive_from_robot()
                    if res:
                        last_data=0
                        if i==9:
                            last_data=1
                        send_res = self.ms.send_joint(curve_js[-1],last_data)
                        curve_js_exe.append(status[2])
                        curve_ts.append(status[1])
                        break
        except KeyboardInterrupt:
            print("Interuptted")
            self.ms.end_stream()
            exit()
        self.ms.end_stream()

        curve_ts=np.array(curve_ts)
        curve_ts=curve_ts-curve_ts[0]
        return np.array(curve_ts),np.array(curve_js_exe)

    def traverse_js_curve(self,curve_js,end_q):

        # start streaming
        self.start_ms()

        # create a quintic polynomial before/after curve to respect jerk,acc,vel constraints
        pre_curve_js=[]
        total_t = 2 # total time, TODO: Make total_t associate with total q
        total_stamp=int(total_t/self.dt)
        for j in range(6):
            v_tar = (curve_js[1][j]-curve_js[0][j])/self.dt
            a_tar = (((curve_js[2][j]-curve_js[1][j])/self.dt)-v_tar)/self.dt
            xs=0
            xe=total_t
            xe1=total_t+self.dt
            xe2=total_t+self.dt*2
            # A=[[xs**5,xs**4,xs**3,xs**2,xs,1],[5*xs**4,4*xs**3,3*xs**2,2*xs,1,0],[20*xs**3,12*xs**2,6*xs,2,0,0],\
            #     [xe**5,xe**4,xe**3,xe**2,xe,1],[5*xe**4,4*xe**3,3*xe**2,2*xe,1,0],[20*xe**3,12*xe**2,6*xe,2,0,0]]
            # b=[self.ms.robot_joint[j],0,0,curve_js[0][j],v_tar,a_tar]
            A=[[xs**5,xs**4,xs**3,xs**2,xs,1],[5*xs**4,4*xs**3,3*xs**2,2*xs,1,0],[20*xs**3,12*xs**2,6*xs,2,0,0],\
                [xe**5,xe**4,xe**3,xe**2,xe,1],[xe1**5,xe1**4,xe1**3,xe1**2,xe1,1],[xe2**5,xe2**4,xe2**3,xe2**2,xe2,1]]
            b=[self.ms.robot_joint[j],0,0,curve_js[0][j],curve_js[1][j],curve_js[2][j]]
            pp=np.matmul(np.linalg.pinv(A),b)
            pre_curve_js.append(np.polyval(pp,np.linspace(0,total_t,total_stamp+1)))
        pre_curve_js=np.array(pre_curve_js).T
        curve_js=np.vstack((pre_curve_js[:-1],curve_js))

        pro_curve_js=[]
        total_t = 1 # total time, TODO: Make total_t associate with total q
        total_stamp=int(total_t/self.dt)
        for j in range(6):
            v_start = (curve_js[-1][j]-curve_js[-2][j])/self.dt
            a_start = (v_start-((curve_js[-2][j]-curve_js[-1][j])/self.dt))/self.dt
            xs=0
            xe=total_t
            xe1=total_t+self.dt
            xe2=total_t+self.dt*2
            xe3=total_t+self.dt*3
            # A=[[xs**5,xs**4,xs**3,xs**2,xs,1],[5*xs**4,4*xs**3,3*xs**2,2*xs,1,0],[20*xs**3,12*xs**2,6*xs,2,0,0],\
            #     [xe**5,xe**4,xe**3,xe**2,xe,1],[5*xe**4,4*xe**3,3*xe**2,2*xe,1,0],[20*xe**3,12*xe**2,6*xe,2,0,0]]
            # b=[curve_js[-1][j],v_start,a_start,curve_js[-1][j]+d_ang,0,0]
            # b=[curve_js[-1][j]+np.sign(v_start)*d_ang,0,0,curve_js[-1][j],v_start,a_start]
            A=[[xs**5,xs**4,xs**3,xs**2,xs,1],[5*xs**4,4*xs**3,3*xs**2,2*xs,1,0],[20*xs**3,12*xs**2,6*xs,2,0,0],\
                [xe**5,xe**4,xe**3,xe**2,xe,1],[xe1**5,xe1**4,xe1**3,xe1**2,xe1,1],[xe2**5,xe2**4,xe2**3,xe2**2,xe2,1]]
            b=[end_q[j],0,0,curve_js[-1][j],curve_js[-2][j],curve_js[-3][j]]
            pp=np.matmul(np.linalg.pinv(A),b)
            pro_curve_js.append(np.flip(np.polyval(pp,np.linspace(0,total_t,total_stamp+1))))
        pro_curve_js=np.array(pro_curve_js).T
        curve_js=np.vstack((curve_js,pro_curve_js[1:]))

        curve_ts,curve_js_exe=self._send_curve(curve_js)
        return curve_ts,curve_js_exe,curve_js
    
    def jog_joint(self,target_q,total_t=4):

        # start streaming
        self.start_ms()

        target_q=np.array(target_q)
        # create a quintic polynomial curve which respects jerk,acc,vel constraints
        d_ang=target_q-self.ms.robot_joint
        # total time, TODO: Make total_t associate with total q
        total_stamp=int(total_t/self.dt)
        curve_js=[]
        for j in range(6):
            xs=0
            xe=total_t
            A=[[xs**5,xs**4,xs**3,xs**2,xs,1],[5*xs**4,4*xs**3,3*xs**2,2*xs,1,0],[20*xs**3,12*xs**2,6*xs,2,0,0],\
                [xe**5,xe**4,xe**3,xe**2,xe,1],[5*xe**4,4*xe**3,3*xe**2,2*xe,1,0],[20*xe**3,12*xe**2,6*xe,2,0,0]]
            b=[self.ms.robot_joint[j],0,0,self.ms.robot_joint[j]+d_ang[j],0,0]
            pp=np.matmul(np.linalg.pinv(A),b)
            curve_js.append(np.polyval(pp,np.linspace(0,total_t,total_stamp+1)))
        curve_js=np.array(curve_js).T

        curve_ts,curve_js_exe=self._send_curve(curve_js)
        return curve_ts,curve_js_exe,curve_js
    
    def logged_data_analysis(self,robot,curve_exe_js,timestamp):
        timestamp=np.array(timestamp).astype(float)*1e-3 # from msec to sec

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

        return lam_exec, curve_exe, curve_exe_R,curve_exe_js_act, act_speed, timestamp
    
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


def main():

    ms=MotionStream()
    # robot
    robot=m710ic(d=50)
    mst = MS_toolbox(ms,robot)

    # curve_js=read_csv('../ilc_fanuc/data/blade/Curve_js.csv',header=None).values
    # mst.jog_joint(np.deg2rad([-53.480,-3.169,-13.7940,-16.4238,-76.9452,169.0873978]))
    curve_js=read_csv('../ilc_fanuc/data/wood/Curve_js.csv',header=None).values
    # mst.jog_joint(np.deg2rad([-39.416,8.816,-3.124,-18.552,-102.542,125.572]))
    curve_ts,curve_js_exe,curve_js_plan=mst.jog_joint(np.deg2rad([ -39.46471413,    8.72574443,   -3.21367673,  -18.61175533,
       -102.45568726,  125.69278819]))
    
    # curve_js=np.array(curve_js)
    # curve_js=curve_js[::50]
    # curve_ts,curve_js_exe,curve_js_plan=mst.traverse_js_curve(curve_js,[ 0.2451047 ,  0.13839685, -0.13633289, -0.53577584, -1.60583352,
    #     0.69042748])

    # curve_exe=[]
    # for i in range(len)
    timestamp_plan = np.linspace(0,(len(curve_js_plan)-1)*mst.dt,len(curve_js_plan))
    plt.plot(timestamp_plan,rad2deg(curve_js_plan[:,0]),label='Joint Plan')
    plt.plot(curve_ts/1000.,rad2deg(curve_js_exe[:,0]),label='Joint Exe')
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()

        