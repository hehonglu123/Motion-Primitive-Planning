import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from MotionSend import *
from utils import *
from lambda_calc import *

ms=MotionSend()
robot=abb6640(d=50)


dataset='from_NX/'
data_dir="../../../train_data/"+dataset
curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values

fitting_output="fitting_output/"+dataset+"threshold0.02/"
df = read_csv(fitting_output+"curve_exe_v193.359375_z10.csv")

##############################train_data analysis#####################################
lam, curve_exe, curve_exe_R,curve_exe_js, speed, timestamp=ms.logged_data_analysis(robot,df)
error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:],extension=True)


start_idx=np.argmin(np.linalg.norm(curve[0,:3]-curve_exe,axis=1))
end_idx=np.argmin(np.linalg.norm(curve[-1,:3]-curve_exe,axis=1))

curve_exe=curve_exe[start_idx:end_idx+1]
curve_exe_R=curve_exe_R[start_idx:end_idx+1]
speed=speed[start_idx:end_idx+1]
lam=calc_lam_cs(curve_exe)

speed=reject_outliers(np.array(speed))


speed[start_idx:end_idx+1]
df=DataFrame({'average speed':[np.average(speed)],'max speed':[np.amax(speed)],'min speed':[np.amin(speed)],'std speed':[np.std(speed)],\
    'average error':[np.average(error)],'max error':[np.max(error)],'min error':[np.amin(error)],'std error':[np.std(error)],\
    'average angle error':[np.average(angle_error)],'max angle error':[max(angle_error)],'min angle error':[np.amin(angle_error)],'std angle error':[np.std(angle_error)]})

df.to_csv(fitting_output+'speed_info.csv',header=True,index=False)