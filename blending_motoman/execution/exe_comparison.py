import numpy as np
from general_robotics_toolbox import *
import sys
sys.path.append('../../toolbox')
from robots_def import *
from MotionSend_motoman import *

def main():
    robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun2.csv',\
        pulse2deg_file_path='../../config/MA2010_A0_pulse2deg.csv',d=50)

    ms = MotionSend(robot)
    # datasets=['movec_smooth','movec_30_car','movec_30_ori','movec+movel_smooth']#,'movel_smooth','movel_30_car','movel_30_ori']
    # datasets=['movel_smooth_fixedR','movel_smooth','movel_30_car','movel_30_ori']
    datasets=['movel_30_car']
    # datasets=['movec_30_car','movec_30_ori','movec+movel_smooth']
    speed=[50,200,400,800,1500]
    # speed=[1500,800,400,200,50]
    zone=[None,0,1,3,5,8]

    for dataset in datasets:
        for v in speed:
            for z in zone: 
                timestamp,curve_exe_js,cmd_num,_=ms.exe_from_file(robot,'../data/'+dataset+"/command.csv",v,z)
       
                if not os.path.exists(dataset):
                   os.makedirs(dataset)

                np.savetxt(dataset+"/curve_exe"+"_"+str(v)+"_"+str(z)+".csv",np.hstack((timestamp.reshape((-1,1)),cmd_num.reshape((-1,1)),curve_exe_js)),delimiter=',',comments='')


if __name__ == "__main__":
    main()