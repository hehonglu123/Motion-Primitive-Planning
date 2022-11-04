import numpy as np
from math import degrees,radians,ceil,floor
import yaml
from copy import deepcopy
from pandas import read_csv,DataFrame
from pathlib import Path
import os
import sys
sys.path.append('../fanuc_toolbox')
from fanuc_utils import *
sys.path.append('../../../greedy_fitting')
from greedy import *

sys.path.append('../../../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from blending import *

# all_objtype=['curve_blade_scale','curve_wood']
all_objtype=['curve_wood']
# all_max_error_thres = [0.02,0.05,0.1]
# all_max_error_thres = [0.05,0.1]
all_max_error_thres = [0.2,0.5]

for obj_type in all_objtype:
    print(obj_type)

    ### data directory
    data_dir='../data/baseline_m10ia/'+obj_type+'/'
    # data_dir='../data/'+obj_type+'/single_arm_de/'
    print(data_dir)

    ## robot
    toolbox_path = '../../../toolbox/'
    robot = robot_obj(toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    
    ## load curve
    curve_js =np.loadtxt(data_dir+"Curve_js.csv",delimiter=',')

    min_length=10

    for max_error_threshold in all_max_error_thres:
        print("Max error thres:",max_error_threshold)
        greedy_fit_obj=greedy_fit(robot,curve_js, min_length=min_length,max_error_threshold=max_error_threshold)

        breakpoints,primitives,p_bp,q_bp=greedy_fit_obj.fit_under_error()

        ############insert initial configuration#################
        primitives.insert(0,'movej_fit')
        p_bp.insert(0,[greedy_fit_obj.curve_fit[0]])
        q_bp.insert(0,[greedy_fit_obj.curve_fit_js[0]])

        ###adjust breakpoint index
        breakpoints[1:]=breakpoints[1:]-1

        print(len(breakpoints))
        print(len(primitives))
        print(len(p_bp))
        print(len(q_bp))

        cmd_dir = data_dir+'greedy_'+str(int(max_error_threshold*1000))+'/'
        Path(cmd_dir).mkdir(exist_ok=True)

        # save motion program commands
        df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'points':p_bp,'q_bp':q_bp})
        df.to_csv(cmd_dir+'/command.csv',header=True,index=False)
