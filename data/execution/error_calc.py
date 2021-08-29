import sys
sys.path.append('../../toolbox')
from error_check import *
from robot_def import *
from pandas import *
import numpy as np



###All in base frame
col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
data = read_csv("../from_interp/Curve_in_base_frame.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_direction_x=data['direction_x'].tolist()
curve_direction_y=data['direction_y'].tolist()
curve_direction_z=data['direction_z'].tolist()
curve_gt=np.vstack((curve_x, curve_y, curve_z)).T


col_names=['X', 'Y', 'Z'] 
data = read_csv("Curve_moveL.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_moveL=np.vstack((curve_x, curve_y, curve_z)).T

###read interpolated curves in joint space
col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv("Curve_moveJ.csv", names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

curve_moveJ=[]
for q in curve_js:
	curve_moveJ.append(1000.*fwd(q).p)

curve_moveJ=np.array(curve_moveJ)

print("moveL max error: ", calc_max_error(curve_moveL,curve_gt), "avg error: ", calc_avg_error(curve_moveL,curve_gt))
print("moveJ max error: ", calc_max_error(curve_moveJ,curve_gt), "avg error: ", calc_avg_error(curve_moveJ,curve_gt))