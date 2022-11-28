import numpy as np
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pandas import *
from pathlib import Path

import sys, yaml
sys.path.append('../fanuc_toolbox')
sys.path.append('../../../greedy_fitting')
sys.path.append('../../../toolbox')
from fitting_toolbox_dual import *
from toolbox_circular_fit import *
from robots_def import *
from general_robotics_toolbox import *
from error_check import *
from fanuc_utils import *
from dual_arm import *
#####################3d curve-fitting with MoveL, MoveJ, MoveC; stepwise incremental bi-section searched self.breakpoints###############################

class greedy_fit(fitting_toolbox):
    def __init__(self,robot1,robot2,curve_js1,curve_js2,min_length,max_error_threshold,max_ori_threshold=np.radians(3)):
        super().__init__(robot1,robot2,curve_js1,curve_js2)
        self.max_error_threshold=max_error_threshold
        self.max_ori_threshold=max_ori_threshold
        self.step=int(len(curve_js1)/25)

        self.slope_constraint=np.radians(180)
        self.break_early=False
        ###initial primitive candidates
        self.primitives={'movel_fit':self.movel_fit,'movej_fit':self.movej_fit,'movec_fit':self.movec_fit}

    def update_dict(self,curve_js1,curve_relative,curve_relative_R):
        ###form new error dict
        error_dict={}
        ori_error_dict={}
        curve_fit_dict={}
        curve_fit_R_dict={}

        ###fit all 3 for all robots first
        for key in self.primitives:
            if 'movej' in key:
                curve_fit_dict[key],curve_fit_R_dict[key],_,p_error,ori_error=self.primitives[key](curve_relative,curve_js1,curve_relative_R,self.robot1,self.curve_fit_js1[-1] if len(self.curve_fit_js1)>0 else [])
            else:
                curve_fit_dict[key],curve_fit_R_dict[key],_,p_error,ori_error=self.primitives[key](curve_relative,curve_js1,curve_relative_R,self.robot1,self.curve_fit1[-1] if len(self.curve_fit1)>0 else [],self.curve_fit_R1[-1] if len(self.curve_fit_R1)>0 else [])

            error_dict[key]=p_error
            ori_error_dict[key]=ori_error
        

        return error_dict,ori_error_dict,curve_fit_dict,curve_fit_R_dict

    def bisect(self,cur_idx):

        next_point = min(self.step,len(self.curve_js1)-self.breakpoints[-1])
        prev_point=0
        prev_possible_point=0


        while True:
            ###end condition
            if next_point==prev_point:
                ###TODO: may not be the same comb with min value
                if min(error_dict.values())<self.max_error_threshold and min(ori_error_dict.values())<self.max_ori_threshold:
                    ##find min comb
                    primitive1=min(error_dict, key=error_dict.get)
                    print('min relative error: ',min(error_dict.values()))
                    return primitive1,curve_fit_dict[primitive1],curve_fit_R_dict[primitive1]

                else:
                    next_point=max(prev_possible_point,2)
                    indices=range(cur_idx,cur_idx+next_point)
                    error_dict,ori_error_dict,curve_fit_dict,curve_fit_R_dict=\
                        self.update_dict(self.curve_js1[indices],self.relative_path[indices],self.relative_R[indices])
                    ##find min comb
                    primitive1=min(error_dict, key=error_dict.get)
                    print('min relative error: ',min(error_dict.values()))
                    return primitive1,curve_fit_dict[primitive1],curve_fit_R_dict[primitive1]

            ###fitting
            indices=range(cur_idx,cur_idx+next_point)
            error_dict,ori_error_dict,curve_fit_dict,curve_fit_R_dict=\
                self.update_dict(self.curve_js1[indices],self.relative_path[indices],self.relative_R[indices])

            ###bp going backward to meet threshold
            if min(error_dict.values())>self.max_error_threshold or min(ori_error_dict.values())>self.max_ori_threshold:
                prev_point_temp=next_point
                next_point-=int(np.abs(next_point-prev_point)/2)
                prev_point=prev_point_temp

            ###bp going forward to get close to threshold
            else:
                prev_possible_point=next_point
                prev_point_temp=next_point
                next_point= min(next_point + int(np.abs(next_point-prev_point)),len(self.curve_js1)-cur_idx)
                prev_point=prev_point_temp


    def fit_under_error(self):

        ###initialize
        self.breakpoints=[0]
        primitives_choices1=[]
        points1=[]
        q_bp1=[]
        primitives_choices2=[]
        points2=[]
        q_bp2=[]

        self.curve_fit1=[]
        self.curve_fit_R1=[]
        self.curve_fit_js1=[]
        self.curve_fit2=[]
        self.curve_fit_R2=[]
        self.curve_fit_js2=[]

        while self.breakpoints[-1]<len(self.relative_path)-1:
            
            # max_errors1={}
            # max_ori_errors1={}
            # length1={}
            # curve_fit1={}
            # curve_fit_R1={}
            # curve_fit_js1={}

            ###bisection search for each primitive 
            ###TODO: pass curve_js from j fit
            primitive1,curve_fit1,curve_fit_R1=self.bisect(self.breakpoints[-1])

            ###convert relative curve_fit into world frame, then solves inv
            curve_fit1_world=copy.deepcopy(curve_fit1)
            curve_fit_R1_world=copy.deepcopy(curve_fit_R1)

            # curve_fit2=[]
            # curve_fit_js2=[]
            for i in range(len(curve_fit1)):
                pose2_world_now=self.robot2.fwd(self.curve_js2[i+self.breakpoints[-1]],world=True)
                curve_fit1_world[i]=pose2_world_now.p+pose2_world_now.R@curve_fit1[i]
                curve_fit_R1_world[i]=pose2_world_now.R@curve_fit_R1[i]

                # curve_fit_js2.append(self.curve_js2[i+self.breakpoints[-1]])
                # curve_fit2.append(self.robot2.fwd(self.curve_js2[i+self.breakpoints[-1]]).p)

            ###solve inv_kin here
            if len(self.curve_fit_js1)>1:
                curve_fit_js1=car2js(self.robot1,self.curve_fit_js1[-1],curve_fit1_world,curve_fit_R1_world)
            else:
                curve_fit_js1=car2js(self.robot1,self.curve_js1[0],curve_fit1_world,curve_fit_R1_world)
            self.curve_fit_js1.extend(curve_fit_js1)

            ###generate output
            if primitive1=='movec_fit':
                points1.append([curve_fit1[int(len(curve_fit1)/2)],curve_fit1[-1]])
                q_bp1.append([curve_fit_js1[int(len(curve_fit_R1)/2)],curve_fit_js1[-1]])
            elif primitive1=='movel_fit':
                points1.append([curve_fit1[-1]])
                q_bp1.append([curve_fit_js1[-1]])
            else:
                points1.append([curve_fit1[-1]])
                q_bp1.append([curve_fit_js1[-1]])


            self.breakpoints.append(min(self.breakpoints[-1]+len(curve_fit1),len(self.curve1)))
            self.curve_fit1.extend(curve_fit1)
            self.curve_fit_R1.extend(curve_fit_R1)

            primitives_choices1.append(primitive1)
            
            ## robot2 follow robot1 primitive
            ## but always use moveL in robot controller in FANUC's setting
            primitives_choices2.append(primitive1) 
            if primitive1=='movec_fit':
                points2.append([self.curve2[self.breakpoints[-1]-int(len(curve_fit1)/2)],self.curve2[self.breakpoints[-1]-1]])
                q_bp2.append([self.curve_js2[self.breakpoints[-1]-int(len(curve_fit1)/2)],self.curve_js2[self.breakpoints[-1]-1]])
            elif primitive1=='movel_fit':
                points2.append([self.curve2[self.breakpoints[-1]-1]])
                q_bp2.append([self.curve_js2[self.breakpoints[-1]-1]])

            print(self.breakpoints)
            print(primitives_choices1)

        ##############################check error (against fitting back projected curve)##############################

        # max_error,max_error_idx=calc_max_error(self.curve_fit,self.curve_backproj)
        # print('max error: ', max_error)

        self.curve_fit1=np.array(self.curve_fit1)
        self.curve_fit_R1=np.array(self.curve_fit_R1)
        self.curve_fit_js1=np.array(self.curve_fit_js1)


        return np.array(self.breakpoints),primitives_choices1,points1,q_bp1,primitives_choices2,points2,q_bp2

    def merge_bp(self,breakpoints,primitives_choices1,points1,q_bp1,primitives_choices2,points2,q_bp2):
        points1_np=np.array([item[0] for item in points1])
        points2_np=np.array([item[0] for item in points2])
        


def main():
    ###read in points
    dataset='curve_1/'
    # dataset='curve_2_scale/'
    curve_dir='../../../data/'+dataset
    data_dir="../data/"+dataset
    solution_dir=data_dir+'dual_arm_de/'
    
    ## robot
    toolbox_path = '../../../toolbox/'
    robot1 = robot_obj('FANUC_m10ia',toolbox_path+'robot_info/fanuc_m10ia_robot_default_config.yml',tool_file_path=toolbox_path+'tool_info/paintgun.csv',d=50,acc_dict_path=toolbox_path+'robot_info/m10ia_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])
    robot2=robot_obj('FANUC_lrmate200id',toolbox_path+'robot_info/fanuc_lrmate200id_robot_default_config.yml',tool_file_path=solution_dir+'tcp.csv',acc_dict_path=toolbox_path+'robot_info/lrmate200id_acc_compensate.pickle',j_compensation=[1,1,-1,-1,-1,-1])

    relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2=initialize_data(dataset,curve_dir,solution_dir,robot1,robot2)

    min_length=20
    fit_error=0.5
    greedy_fit_obj=greedy_fit(robot1,robot2,curve_js1[::1],curve_js2[::1],min_length,fit_error)

    ###set primitive choices, defaults are all 3
    # greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit,'movec_fit':greedy_fit_obj.movec_fit}
    greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit,'movec_fit':greedy_fit_obj.movec_fit}
    # greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit}

    breakpoints,primitives_choices1,points1,q_bp1,primitives_choices2,points2,q_bp2=greedy_fit_obj.fit_under_error()

    ###plt
    ###3D plot in global frame
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(greedy_fit_obj.curve_fit1[:,0], greedy_fit_obj.curve_fit1[:,1],greedy_fit_obj.curve_fit1[:,2], 'gray', label='arm1')
    plt.legend()
    plt.show()

    ###3D plot in robot2 tool frame
    _,_,_,_,relative_path_fit,relative_path_fit_R=form_relative_path(greedy_fit_obj.curve_fit_js1,greedy_fit_obj.curve_js2,robot1,robot2)
    print(relative_path_fit)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(greedy_fit_obj.relative_path[:,0], greedy_fit_obj.relative_path[:,1],greedy_fit_obj.relative_path[:,2], 'gray', label='original')
    ax.plot3D(relative_path_fit[:,0], relative_path_fit[:,1],relative_path_fit[:,2], 'green', label='fitting')
    plt.legend()
    plt.show()

    ############insert initial configuration#################
    primitives_choices1.insert(0,'movej_fit')
    points1.insert(0,[greedy_fit_obj.curve_fit1[0]])
    q_bp1.insert(0,[greedy_fit_obj.curve_fit_js1[0]])

    primitives_choices2.insert(0,'movej_fit')
    points2.insert(0,[greedy_fit_obj.curve2[0]])
    q_bp2.insert(0,[greedy_fit_obj.curve_js2[0]])

    print(len(breakpoints))
    print(len(primitives_choices1))
    print(len(points1))

    output_dir=solution_dir+'greedy'+str(fit_error)+'/'
    Path(output_dir).mkdir(exist_ok=True)

    breakpoints[1:]=breakpoints[1:]-1
    ###save arm1
    df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices1,'p_bp':points1,'q_bp':q_bp1})
    df.to_csv(output_dir+'command1.csv',header=True,index=False)

    ###save arm2
    df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices2,'p_bp':points2,'q_bp':q_bp2})
    df.to_csv(output_dir+'command2.csv',header=True,index=False)


if __name__ == "__main__":
    main()
