from RobotRaconteur.Client import *
import time, argparse, sys
import numpy as np
from pandas import *
sys.path.append('../../toolbox')
from robot_def import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    ###read actual curve
    col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv("trajectory/arm1.csv", names=col_names)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    curve_js1=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

    col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
    data = read_csv("trajectory/arm2.csv", names=col_names)
    curve_q1=data['q1'].tolist()
    curve_q2=data['q2'].tolist()
    curve_q3=data['q3'].tolist()
    curve_q4=data['q4'].tolist()
    curve_q5=data['q5'].tolist()
    curve_q6=data['q6'].tolist()
    curve_js2=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

    ###second arm base pose
    base2_R=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    base2_p=np.array([6000,0,0])

    relative_path=[]
    for i in range(len(curve_js1)):
        pose1_now=fwd(curve_js1[i])
        pose2_now_world=fwd(curve_js2[i],base2_R,base2_p)
        print(fwd(curve_js2[i]).p)

        relative_path.append(pose1_now.p-pose2_now_world.p)

    relative_path=np.array(relative_path)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(relative_path[:,0], relative_path[:,1], relative_path[:,2], 'gray')

    plt.show()

if __name__ == "__main__":
    main()