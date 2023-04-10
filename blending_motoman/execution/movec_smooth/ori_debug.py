import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../../toolbox')
from robots_def import *
from MotionSend_motoman import *

curve = read_csv('../../data/movec_smooth/Curve_in_base_frame.csv',header=None).values


robot=robot_obj('MA2010_A0',def_path='../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../config/weldgun2.csv',\
	pulse2deg_file_path='../../../config/MA2010_A0_pulse2deg_real.csv',d=50)
ms=MotionSend(robot)

w=[]
data=np.loadtxt('curve_exe_50_0.csv',delimiter=',')
log_results=(data[:,0],data[:,2:8],data[:,1])
lam, curve_exe, curve_exe_R,curve_exe_js, speed_exe, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=False)

# curve_exe_js=np.loadtxt('curve_exe_50_0.csv',delimiter=',')[:,2:]
R_init=robot.fwd(curve_exe_js[0]).R
for q in curve_exe_js[1:]:
	k,theta=R2rot(R_init.T@robot.fwd(q).R)
	if len(k*theta)==0:
		continue
	w.append(k*theta)


w=np.array(w)
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(w[:,0], w[:,1], w[:,2], c='gray',label='orientation')
plt.title('MoveC Orientation ktheta')
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim([700,1700])
ax.set_ylim([-500,500])
ax.set_zlim([299,301])

ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='gray',label='exe')
ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='red',label='original')
plt.legend()
plt.title('MoveC Trajectory')
plt.show()