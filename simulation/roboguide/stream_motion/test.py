import socket
from copy import deepcopy
from struct import *
import time
from tkinter.messagebox import showerror
import numpy as np
from numpy import deg2rad,rad2deg
from matplotlib import pyplot as plt

import sys
sys.path.append('../../../toolbox')
from robots_def import *

def fanuc2usual(q):
    q[2]=q[1]+q[2] #j3_usual=j2_fanuc+j3_fanuc
    return q
def usual2fanuc(q):
    q[2]=q[2]-q[1] #j3_fanuc=j3_usual-j2_fanuc
    return q

# get binary function
getbinary = lambda x, n: format(x, 'b').zfill(n)

UDP_IP='127.0.0.2'
UDP_PORT=60015

serverAddr=(UDP_IP,UDP_PORT)
sock=socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
sock.settimeout(10.0)

# robot
robot=m710ic(d=50)

# data, addr = sock.recvfrom(132)
# exit()

# stop packet to reset
end_pkt = pack('>2I',2,1)
sock.sendto(end_pkt, serverAddr)

# initial packet
init_pkt = pack('>2I',0,1)
sock.sendto(init_pkt, serverAddr)

# robot ready
robot_ready=False
# show message once
show_ready=False

time.sleep(0.5)
# robot pre execution stage
while True:
    try:
        data, addr = sock.recvfrom(132) # a status package is 132 byte
        data_unpack=unpack('>3I2B3HI27f',data)
        # print(data_unpack)
        # exit()
        if not show_ready:
            print("Please Press Start")
            show_ready=True

        status_bit=getbinary(data_unpack[3],4)
        if status_bit[3] == '1' and status_bit[1] == '1':
            print("Robot is ready to move.")
            break
    except KeyboardInterrupt:
        print("Interuptted")
        exit()

curve_js_start=fanuc2usual(deg2rad(data_unpack[18:24]))
curve_js=[curve_js_start]

dt=0.008
vd=0.01 # 1deg/sec
target_q=10 # move 10 deg
# total_stamp=int(target_q/vd/dt)
total_stamp=200

this_vel=np.zeros(6)
this_acc=np.zeros(6)

for i in range(4):
    # jerk_ctrl=-1*(-1**i)*deepcopy(robot.joint_jrk_limit/1000.)
    jerk_ctrl=(-1**i)*deepcopy(robot.joint_jrk_limit/1000.)
    for t in range(total_stamp):
        curve_js.append(curve_js[-1]+dt*this_vel)
        this_vel += this_acc*dt
        this_vel=np.clip(this_vel,-robot.joint_vel_limit*0.9,robot.joint_vel_limit*0.9)
        this_acc += jerk_ctrl*dt
        this_acc=np.clip(this_acc,-robot.joint_acc_limit*0.9,robot.joint_acc_limit*0.9)

        # next_x = np.polyval([(1./6)*jerk_ctrl,(1./2)*this_acc,this_vel,curve_js[-1]],dt)
        # next_x=np.clip(next_x,robot.lower_limit,robot.upper_limit)
        # curve_js.append(next_x)
        # this_vel += np.polyval([(1./2)*jerk_ctrl,this_acc,0],dt)
        # this_vel=np.clip(this_vel,-robot.joint_vel_limit*0.9,robot.joint_vel_limit*0.9)
        # this_acc += jerk_ctrl*dt
        # this_acc=np.clip(this_acc,-robot.joint_acc_limit*0.9,robot.joint_acc_limit*0.9)

print(curve_js[:20])

curve_js_exe=[]
seq_num=data_unpack[2]
SEND_JOINT=True
st=0
all_dt=[]
last_t_exe=data_unpack[8]
all_dt_exe=[]
print(len(curve_js))
# robot execution stage
for i in range(len(curve_js)):
    try:
        last_data=0
        if i==len(curve_js)-1:
            print("Set last data to 1")
            last_data=1
        
        send_joint=1
        if not SEND_JOINT:
            send_joint=0

        cmd_q=usual2fanuc(rad2deg(curve_js[i]))
        # send motion package
        cmd_pkt = pack('>3I2B2H2B4H9f',1,data_unpack[1],seq_num,last_data,0,0,0,send_joint,0,0,0,0,0,\
            cmd_q[0],cmd_q[1],cmd_q[2],cmd_q[3],cmd_q[4],cmd_q[5],0,0,0)
        
        
        et = time.perf_counter()
        all_dt.append(et-st)
        if all_dt[-1]<dt-0.5:
            time.sleep(dt-0.5-all_dt[-1])

        # send pkt
        sock.sendto(cmd_pkt, serverAddr)
        st = time.perf_counter()
        data, addr = sock.recvfrom(132) # a status package is 132 byte
        
        
        data_unpack=unpack('>3I2B3HI27f',data)
        exe_q = fanuc2usual(np.array(data_unpack[18:24]))
        curve_js_exe.append(deg2rad(exe_q))

        all_dt_exe.append(data_unpack[8]-last_t_exe)
        last_t_exe=data_unpack[8]

        status_bit=getbinary(data_unpack[3],4)
        if status_bit[3] == '0':
            print("Robot is stopped.")
            break

        seq_num += 1
    except KeyboardInterrupt:
        print("Interuptted")
        exit()

print("Send all cmd")
# robot ending stage
while True:
    try:
        data, addr = sock.recvfrom(132) # a status package is 132 byte
        data_unpack=unpack('>3I2B3HI27f',data)

        status_bit=getbinary(data_unpack[3],4)
        if status_bit[3] == '0':
            print("Robot is stopped.")
            break
    except KeyboardInterrupt:
        print("Interuptted")
        exit()

# ending stop packet
end_pkt = pack('>2I',2,1)
sock.sendto(end_pkt, serverAddr)

curve_js=np.array(curve_js)
curve_js_exe=np.array(curve_js_exe)

plt.plot(rad2deg(curve_js[:,0]))
plt.plot(rad2deg(curve_js_exe[:,0]))
plt.show()

all_dt_exe=np.array(all_dt_exe)/1000.
plt.plot(all_dt[1:])
plt.plot(all_dt_exe)
plt.show()