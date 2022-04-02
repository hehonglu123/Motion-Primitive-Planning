import numpy as np
from pandas import *
import sys, traceback, time
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time


sys.path.append('../../toolbox')
from robots_def import *
from utils import *

col_names=['X', 'Y', 'Z','direction_x','direction_y','direction_z'] 
data = read_csv("../from_ge/Curve_in_base_frame2.csv", names=col_names)
curve_x=data['X'].tolist()
curve_y=data['Y'].tolist()
curve_z=data['Z'].tolist()
curve_direction_x=data['direction_x'].tolist()
curve_direction_y=data['direction_y'].tolist()
curve_direction_z=data['direction_z'].tolist()
curve=np.vstack((curve_x, curve_y, curve_z)).T
curve_normal=np.vstack((curve_direction_x, curve_direction_y, curve_direction_z)).T


robot1 = abb6640(d=50)

theta = 4.00553063332701*np.ones(len(curve))

start_time = time.time()

for i in range(len(curve)):
    if i==0:
        R_temp=direction2R(curve_normal[i],-curve[i+1]+curve[i])
        R=np.dot(R_temp,Rz(theta[i]))
        try:
            q_out=[robot1.inv(curve[i],R)[0]]
        except:
            traceback.print_exc()
            #return 999

    else:
        R_temp=direction2R(curve_normal[i],-curve[i]+curve[i-1])
        R=np.dot(R_temp,Rz(theta[i]))
        try:
 			###get closet config to previous one
            q_inv_all=robot1.inv(curve[i],R)
            temp_q=q_inv_all-q_out[-1]
            order=np.argsort(np.linalg.norm(temp_q,axis=1))
            q_out.append(q_inv_all[order[0]])
        except:
            traceback.print_exc()
            #return 999
q_out = np.array(q_out)

    
    
print("--- %s seconds ---" % (time.time() - start_time))
###output to csv
df=DataFrame({'q0':q_out[:,0],'q1':q_out[:,1],'q2':q_out[:,2],'q3':q_out[:,3],'q4':q_out[:,4],'q5':q_out[:,5]})
df.to_csv('qbestcurve_new.csv',header=False,index=False)