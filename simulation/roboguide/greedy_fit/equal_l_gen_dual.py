import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../../../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *
from pathlib import Path

all_objtype=['curve_2_scale','curve_1']
# all_objtype=['curve_1']

num_ls=[25,30,50,100]

for obj_type in all_objtype:

    print(obj_type)
    data_dir='../data/'+obj_type+'/dual_arm_de/'
    print(data_dir)

    ## robot
    toolbox_path = '../../../toolbox/'
    robot1 = robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    robot2=robot_obj('FANUC_lrmate200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path=data_dir+'tcp.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc.pickle',j_compensation=[1,1,-1,-1,-1,-1])

    curve_js1 = read_csv(data_dir+'arm1.csv',header=None).values
    curve_js2 = read_csv(data_dir+'arm2.csv',header=None).values

    ## cartesian in robot's own user frame
    curve1 = []
    for q in curve_js1:
        curve1.append(robot1.fwd(q).p)
    curve1=np.array(curve1)

    curve2 = []
    for q in curve_js2:
        curve2.append(robot2.fwd(q).p)
    curve2=np.array(curve2)

    for num_l in num_ls:
        Path(data_dir+str(num_l)).mkdir(exist_ok=True)

        breakpoints1=np.linspace(0,len(curve_js1),num_l+1).astype(int)
        breakpoints2=np.linspace(0,len(curve_js2),num_l+1).astype(int)
        
        points1=[]
        points1.append([np.array(curve1[0,:3])])
        points2=[]
        points2.append([np.array(curve2[0,:3])])
        q_bp1=[]
        q_bp1.append([np.array(curve_js1[0])])
        q_bp2=[]
        q_bp2.append([np.array(curve_js2[0])])

        for i in range(1,num_l+1):
            points1.append([np.array(curve1[breakpoints1[i]-1,:3])])
            q_bp1.append([np.array(curve_js1[breakpoints1[i]-1])])
            points2.append([np.array(curve2[breakpoints2[i]-1,:3])])
            q_bp2.append([np.array(curve_js2[breakpoints2[i]-1])])
            
        primitives_choices1=['movej_fit']+['movel_fit']*num_l
        primitives_choices2=['movej_fit']+['movel_fit']*num_l

        df=DataFrame({'breakpoints':breakpoints1,'primitives':primitives_choices1,'points':points1,'q_bp':q_bp1})
        df.to_csv(data_dir+str(num_l)+'/command1.csv',header=True,index=False)
        df=DataFrame({'breakpoints':breakpoints2,'primitives':primitives_choices2,'points':points2,'q_bp':q_bp2})
        df.to_csv(data_dir+str(num_l)+'/command2.csv',header=True,index=False)