import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../../../../toolbox')
sys.path.append('../')
from exe_comparison import *
from robots_def import *
from error_check import *


def main():
    ms = MotionSend()
    col_names=['X', 'Y', 'Z','direction_x', 'direction_y', 'direction_z'] 
    data = read_csv("../../../../data/from_ge/Curve_in_base_frame2.csv", names=col_names)
    curve_x=data['X'].tolist()
    curve_y=data['Y'].tolist()
    curve_z=data['Z'].tolist()
    curve=np.vstack((curve_x, curve_y, curve_z)).T



    vmax = speeddata(10000,9999999,9999999,999999)
    speed={"vmax":vmax}
    zone={"z10":z10}

    for s in speed:
        for z in zone: 
            primitives=['movel_fit']*len(curve)
            breakpoints=np.linspace(0,len(curve),num=len(curve),endpoint=False,dtype=int)
            points_list=curve
            filename_js='../../../../data/from_ge/Curve_js2.csv'
            curve_exe_js=ms.exec_motions(primitives,breakpoints,points_list,filename_js,speed[s],zone[z])
   

            f = open("dense_curve_exe"+"_"+s+"_"+z+".csv", "w")
            f.write(curve_exe_js)
            f.close()

if __name__ == "__main__":
    main()