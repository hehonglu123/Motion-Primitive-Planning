from math import radians,pi
import numpy as np
from pandas import *
import yaml
from matplotlib import pyplot as plt
from pathlib import Path

from general_robotics_toolbox import *
from general_robotics_toolbox.general_robotics_toolbox_invkin import *
import sys


# from simulation.roboguide.fanuc_toolbox.fanuc_client import FANUCClient, TPMotionProgram, joint2robtarget, jointtarget, robtarget
# from toolbox.robots_def import arb_robot, m900ia
sys.path.append('../../../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *
sys.path.append('../../../greedy_fitting')
from greedy import *

# all_objtype=['wood','blade_scale']
# all_objtype=['curve_blade_scale']
all_objtype=['curve_wood']
# all_objtype=['blade']
# all_objtype=['wood']

# num_ls=[80,100,150]
num_ls=[25,30,50,100]

for obj_type in all_objtype:

    # obj_type='wood'
    # obj_type='blade'
    print(obj_type)
    
    # data_dir='../data/baseline_m10ia/'+obj_type+'/'
    data_dir='../data/'+obj_type+'/single_arm_de/'
    print(data_dir)

    # robot=m10ia(d=50)
    toolbox_path = '../../../toolbox/'
    robot = robot_obj(toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    # curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    ###read actual curve
    curve_js = read_csv(data_dir+"Curve_js.csv",header=None).values
    curve_js=np.array(curve_js)
    curve = read_csv(data_dir+"Curve_in_base_frame.csv",header=None).values
    curve = np.array(curve)

    print(robot.fwd(curve_js[0]))
    print(curve[0])
    
    # df=DataFrame({'q0':curve_js[:,0],'q1':curve_js[:,1],'q2':-1*curve_js[:,2],'q3':-1*curve_js[:,3],'q4':-1*curve_js[:,4],'q5':-1*curve_js[:,5]})
    # df.to_csv(data_dir+"Curve_js_comp.csv",header=False,index=False)
    # exit()

    for num_l in num_ls:

        Path(data_dir+str(num_l)).mkdir(exist_ok=True)

        step=int((len(curve_js)-1)/num_l)

        breakpoints = [0]
        primitives = ['movej_fit']
        q_bp = [[curve_js[0]]]
        p_bp = [[robot.fwd(curve_js[0]).p]]
        for i in range(step,len(curve_js),step):
            breakpoints.append(i)
            primitives.append('movel_fit')
            q_bp.append([curve_js[i]])
            p_bp.append([robot.fwd(curve_js[i]).p])
        
        # save motion program commands
        df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'points':p_bp,'q_bp':q_bp})
        df.to_csv(data_dir+str(num_l)+'/command.csv',header=True,index=False)

        # breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
        # breakpoints[1:]=breakpoints[1:]-1

        # primitives_choices=['movej_fit']
        # points=[curve_js[0]]
        # for i in breakpoints[1:]:
        #     primitives_choices.append('movel_fit')
        #     points.append(curve_js[i])
        # points=np.array(points)
        
        # ## save commands
        # df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,\
        #     'J1':points[:,0],'J2':points[:,1],\
        #     'J3':points[:,2],'J4':points[:,3],\
        #     'J5':points[:,4],'J6':points[:,5]})
        # df.to_csv(data_dir+str(num_l)+'/command.csv')
        # df=DataFrame()