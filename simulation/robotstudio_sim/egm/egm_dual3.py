import numpy as np
from general_robotics_toolbox import *
import sys, threading

sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from lambda_calc import *
sys.path.append('../../../toolbox/egm_toolbox')
from EGM_toolbox import *
import rpi_abb_irc5

data_dir='../../../data/wood/'

robot1=abb6640(d=50)
with open(data_dir+'dual_arm/abb1200.yaml') as file:
	H_1200 = np.array(yaml.safe_load(file)['H'],dtype=np.float64)

base2_R=H_1200[:3,:3]
base2_p=1000*H_1200[:-1,-1]

with open(data_dir+'dual_arm/tcp.yaml') as file:
	H_tcp = np.array(yaml.safe_load(file)['H'],dtype=np.float64)
robot2=abb1200(R_tool=H_tcp[:3,:3],p_tool=H_tcp[:-1,-1])

egm1 = rpi_abb_irc5.EGM(port=6510)
egm2 = rpi_abb_irc5.EGM(port=6511)

et1=EGM_toolbox(egm1,robot1)
et2=EGM_toolbox(egm2,robot2)

curve_js1=read_csv(data_dir+"dual_arm/arm1.csv",header=None).values
curve_js2=read_csv(data_dir+"dual_arm/arm2.csv",header=None).values
relative_path=read_csv(data_dir+"Curve_dense.csv",header=None).values

vd=500
idx=et1.downsample24ms(relative_path,vd)

curve_cmd_js1=curve_js1[idx]
curve_cmd_js2=curve_js2[idx]
curve_js1_d=curve_js1[idx]
curve_js2_d=curve_js2[idx]

###jog both arm to start pose
et1.jog_joint(curve_js1[0])
et2.jog_joint(curve_js2[0])

def func1():
	time.sleep(0.001)
	global et1, curve_cmd_js1,timestamp1,curve_exe_js1
	timestamp1,curve_exe_js1=et1.traverse_curve_js(curve_cmd_js1)
	print(time.time())
	print('1 done')

def func2():
	
	global et2, curve_cmd_js2,timestamp2,curve_exe_js2
	timestamp2,curve_exe_js2=et2.traverse_curve_js(curve_cmd_js2)
	print(time.time())
	print('2 done')

def main():
	global et1, curve_cmd_js1,timestamp1,curve_exe_js1,et2, curve_cmd_js2,timestamp2,curve_exe_js2
	p1 = threading.Thread(target=func1)
	p2 = threading.Thread(target=func2)
	p1.start();p2.start()
	time.sleep(5)
	print(timestamp1[-1]-timestamp1[0])
	print(timestamp2[-1]-timestamp2[0])
	plt.plot(timestamp1)
	plt.show()
	plt.plot(timestamp2)
	plt.show()

if __name__ == '__main__':
	main()