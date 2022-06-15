import numpy as np
import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from MotionSend import *

robot=abb6640(d=50)

dataset="../data/wood/"
# data_dir=dataset+'baseline/100L/realrobot/'
data_dir='../ILC/recorded_data/curve1_250/realrobot/'


curve = read_csv(dataset+"Curve_in_base_frame.csv",header=None).values
ms=MotionSend()

s=250
# s=123.046875

curve_exe_all=[]
curve_exe_js_all=[]
timestamp_all=[]
###read in curve_exe
for i in range(5):
    ##read in curve_exe
    # df = read_csv(data_dir+"v"+str(s)+'_iteration'+str(i)+'.csv')
    df = read_csv(data_dir+str(s)+'_iteration'+str(i)+'.csv')
    ##############################data analysis#####################################
    lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df,realrobot=True)
    
    timestamp=timestamp-timestamp[0]

    curve_exe_all.append(curve_exe)
    curve_exe_js_all.append(curve_exe_js)
    timestamp_all.append(timestamp)


    # ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c=np.random.rand(3,),label=str(i+1)+'th trajectory')

###infer average curve from linear interplateion
curve_js_all_new, avg_curve_js, timestamp_d=average_curve(curve_exe_js_all,timestamp_all)
###calculat error with average curve
lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp_d,avg_curve_js)
#############################chop extension off##################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.chop_extension(curve_exe, curve_exe_R,curve_exe_js, speed, timestamp_d,curve[0,:3],curve[-1,:3])

error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
###error plot

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(lam,speed, 'g-', label='Speed')
ax2.plot(lam, error, 'b-',label='Error')
ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Baseline "+dataset+" Speed: "+str(s))
ax1.legend(loc=0)

ax2.legend(loc=0)

plt.legend()
plt.savefig(data_dir+'error_speed.jpg')

plt.show()

###save speed info
# df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
#             'average error':[np.average(error)],'max error':[np.max(error)],'min error':[np.amin(error)],'std error':[np.std(error)],\
#             'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

# df.to_csv(data_dir+'speed_info.csv',header=True,index=False)