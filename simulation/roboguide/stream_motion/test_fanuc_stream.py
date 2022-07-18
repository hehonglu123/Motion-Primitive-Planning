from copy import deepcopy
import time
import numpy as np
from numpy import deg2rad,rad2deg
from matplotlib import pyplot as plt
from fanuc_stream import *
from urllib.request import urlopen 

import sys
sys.path.append('../../../toolbox')
from robots_def import *

# robot
robot=m710ic(d=50)

ms=MotionStream()
motion_url='http://'+ms.robot_ip+'/kcl/set%20port%20dout%20[102]%20=%20on'
res = urlopen(motion_url)
# send an end packet to reset
ms.end_stream()
# clear queue
ms.clear_queue()
# send start packet
ms.start_stream()

# robot pre execution stage
try:
    while True:
        while True:
            res,status = ms.receive_from_robot()
            if res:
                break
        if ms.robot_ready:
            break
except KeyboardInterrupt:
    print("Interuptted")
    ms.end_stream()
    exit()


curve_js_start=deepcopy(status[2])
# create a curve which respects jerk,acc,vel constraints
dt=0.008
d_ang=deg2rad(20) # move 10 deg
total_t = 1 # total time
total_stamp=int(total_t/dt)
curve_js=[]
for j in range(6):
    xs=0
    xe=total_t
    A=[[xs**5,xs**4,xs**3,xs**2,xs,1],[5*xs**4,4*xs**3,3*xs**2,2*xs,1,0],[20*xs**3,12*xs**2,6*xs,2,0,0],\
        [xe**5,xe**4,xe**3,xe**2,xe,1],[5*xe**4,4*xe**3,3*xe**2,2*xe,1,0],[20*xe**3,12*xe**2,6*xe,2,0,0]]
    b=[curve_js_start[j],0,0,curve_js_start[j]+d_ang,0,0]
    pp=np.matmul(np.linalg.pinv(A),b)
    curve_js.append(np.polyval(pp,np.linspace(0,total_t,total_stamp+1)))
curve_js=np.array(curve_js).T

# send cmd
ms.clear_queue()
curve_js_exe=[]
try:
    for i in range(len(curve_js)):
        while True:
            res,status = ms.receive_from_robot()
            if res:
                send_res = ms.send_joint(curve_js[i])
                curve_js_exe.append(status[2])
                break
    for i in range(10): # 50 ms delay
        while True:
            res,status = ms.receive_from_robot()
            if res:
                last_data=0
                if i==9:
                    last_data=1
                send_res = ms.send_joint(curve_js[-1],last_data)
                curve_js_exe.append(status[2])
                break
except KeyboardInterrupt:
    print("Interuptted")
    ms.end_stream()
    exit()
ms.end_stream()

# plot
curve_js_exe=np.array(curve_js_exe)

plt.plot(rad2deg(curve_js[:,0]))
plt.plot(rad2deg(curve_js_exe[:,0]))
plt.show()