from matplotlib.pyplot import contour
import numpy as np
from pandas import *
import general_robotics_toolbox as rox
from random import randint, sample

import sys
sys.path.append('../../toolbox')
from robots_def import *

# parameters
# total_data_num = 30000
total_data_num = 3
point_range = [500,10000]
trans_x_diff = [-500,500]
trans_x_diff = [-500,500]
trans_z_diff = [-100,1000]
# rot_x_diff = [-45,45]
# rot_y_diff = [-45,45]
# rot_z_diff = [-45,45]
rot_ang_diff = [-45,45]

# load curve
col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
data = read_csv("../../data/from_ge/Curve_backproj_js.csv", names=col_names)
curve_q1=data['q1'].tolist()
curve_q2=data['q2'].tolist()
curve_q3=data['q3'].tolist()
curve_q4=data['q4'].tolist()
curve_q5=data['q5'].tolist()
curve_q6=data['q6'].tolist()
curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T
robot=abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

curve_center_id = 32684
cur_center_p = robot.fwd(curve_js[curve_center_id])
curve_trans = rox.Transform(np.eye(3))

# data acquisition
data_cnt = 0
total_len = len(curve_js)
print(total_len)
while data_cnt < total_data_num:
    start_point_id = randint(0,total_len)
    direction = sample([-1,1],k=1)[0]

    # find two goal points
    pt_id = [start_point_id]
    cont_flag = False
    for pt in range(2):
        up_limit = np.max([(0-pt_id[pt])/direction,(total_len-pt_id[pt])/direction])
        if up_limit < point_range[0]:
            cont_flag=True
            break
        pt_id.append(pt_id[pt] + direction*randint(point_range[0],point_range[1]))
    if cont_flag:
        continue
