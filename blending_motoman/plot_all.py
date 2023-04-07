import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
sys.path.append('../toolbox')
from MotionSend_motoman import *
from utils import *
from robots_def import *
from lambda_calc import *


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
    pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=50)
dataset='movec_30_car'

###read in original curve
curve = read_csv('data/'+dataset+'/Curve_in_base_frame.csv',header=None).values
curve_js = read_csv('data/'+dataset+'/Curve_js.csv',header=None).values

###get breakpoints location
# data = read_csv('data/'+dataset+'/command.csv')
# breakpoints=np.array(data['breakpoints'].tolist())
# breakpoints[1:]=breakpoints[1:]-1


data_dir='execution/'+dataset
speed=[50,200,400,800,1500]
zone=[None,0,1,3,5,8]

ms=MotionSend(robot)
for v in speed:
	for z in zone:
		data=np.loadtxt(data_dir+'/curve_exe_'+str(v)+'_'+str(z)+'.csv',delimiter=',')
		log_results=(data[:,0],data[:,2:8],data[:,1])
		lam, curve_exe, curve_exe_R,curve_exe_js, speed_exe, timestamp=ms.logged_data_analysis(robot,log_results,realrobot=False)

		###plot original curve
		plt.figure()
		plt.title('3D Trajectory')
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
		# ax.scatter3D(curve[breakpoints,0], curve[breakpoints,1],curve[breakpoints,2], 'blue')
		#plot execution curve
		ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='execution')
		

		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam,speed_exe, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Cartesian Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		ax1.legend(loc=0)
		ax2.legend(loc=0)
		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error (mm/deg)', color='b')
		ax2.set_ylim(0,4)

		plt.title(dataset+' v_'+str(v)+' z_'+str(z))
		plt.savefig(data_dir+'/curve_exe_'+str(v)+'_'+str(z))
		plt.show()
		