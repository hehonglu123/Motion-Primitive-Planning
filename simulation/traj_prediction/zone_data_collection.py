import numpy as np
from math import pi, cos, sin, radians
import time
import general_robotics_toolbox as rox
from general_robotics_toolbox import general_robotics_toolbox_invkin as roxinv

from abb_motion_program_exec_client import *

client = MotionProgramExecClient()
ENDWAITTIME = 0.1

def send_movels(p1,q1,cf1,p2,q2,cf2,p3,q3,cf3,vel,zone):
    
    robt1 = robtarget(p1, q1, confdata(cf1[0],cf1[1],cf1[2],cf1[3]),[9E+09]*6)
    robt2 = robtarget(p2, q2, confdata(cf2[0],cf2[1],cf2[2],cf2[3]),[9E+09]*6) 
    robt3 = robtarget(p3, q3, confdata(cf3[0],cf3[1],cf3[2],cf3[3]),[9E+09]*6) 

    # move to the start pose and wait
    mp = MotionProgram()
    mp.MoveJ(robt1,v500,fine)
    client.execute_motion_program(mp)
    time.sleep(0.1)

    # the trajectory
    mp = MotionProgram()
    mp.MoveL(robt2,vel,zone)
    mp.MoveL(robt3,vel,fine)
    mp.WaitTime(ENDWAITTIME)
    log_results = client.execute_motion_program(mp)
    
    return log_results

def send_to_pose(p1,q1,cf1):

    robt = robtarget(p1, q1, confdata(cf1[0],cf1[1],cf1[2],cf1[3]),[9E+09]*6)
    mp = MotionProgram()
    mp.MoveJ(robt,v500,fine)
    mp.WaitTime(ENDWAITTIME)
    log_results = client.execute_motion_program(mp)

    return log_results


# configuration of the test
# start_p = np.array([2300,1000,600])
# end_p = np.array([1300,-1000,600])
start_p = np.array([2300,1000,600])
end_p = np.array([1300,-1000,600])
all_Rq = rox.R2q(rox.rot([0,1,0],pi/2))
conf = [0,0,0,1]
side_l = 200
x_divided = 11
step_x = (end_p[0]-start_p[0])/(x_divided-1)
y_divided = 11
step_y = (end_p[1]-start_p[1])/(y_divided-1)
all_vel = [v50,v500]
angels = [90,120,150]

# send_to_pose(start_p,all_Rq,conf)

pcnt = 0 # starting index
start_vi = 1
start_xi = 7
start_yi = 4

st = time.perf_counter()

for vi in range(start_vi,len(all_vel)):
    start_vi = 0
    v_profile = all_vel[vi]

    # from x start to end, and remeber to include the last x

    xcnt = start_xi
    for pxi in range(start_xi,x_divided):
        start_xi = 0
        px = start_p[0]+step_x*pxi

        # from y start to end, and remeber to include the last x
        ycnt = start_yi
        for pyi in range(start_yi,y_divided):
            start_yi = 0
            py = start_p[1]+step_y*pyi
            print(xcnt,ycnt)
            print(px,py)
            # different angel between moveLs
            for ang in angels:
                # move_start = [px-side_l*cos(radians(ang/2)),py+side_l*sin(radians(ang/2)),start_p[2]]
                # move_end = [px-side_l*cos(radians(ang/2)),py-side_l*sin(radians(ang/2)),start_p[2]]
                move_start = [px,py+side_l*sin(radians(ang/2)),start_p[2]-side_l*cos(radians(ang/2))]
                move_end = [px,py-side_l*sin(radians(ang/2)),start_p[2]-side_l*cos(radians(ang/2))]
                move_mid = [px,py,start_p[2]]
                print(move_start)
                print(move_end)
                # send_to_pose(move_start,all_Rq,conf)
                log = send_movels(move_start,all_Rq,conf,move_mid,all_Rq,conf,move_end,all_Rq,conf,v_profile,z10)

                with open("data_param_vertical/log_"+str(v_profile.v_tcp)+"_"+"{:02d}".format(xcnt)+"_"+"{:02d}".format(ycnt)+"_"+\
                            str(ang)+".csv","wb") as f:
                    f.write(log)

            ycnt +=1
            print("Height = 0,Progress:","{:.2f}".format((xcnt*y_divided+ycnt)/(x_divided*y_divided)*100/2),'%. Total Time:',time.strftime("%H:%M:%S", time.gmtime(time.perf_counter()-st)))
            print("=======================================")

            pcnt += 1
        xcnt += 1

# for vi in range(start_vi,len(all_vel)):
#     start_vi = 0
#     v_profile = all_vel[vi]

#     # from x start to end, and remeber to include the last x

#     xcnt = start_xi
#     for pxi in range(start_xi,x_divided):
#         start_xi = 0
#         px = start_p[0]+step_x*pxi

#         # from y start to end, and remeber to include the last x
#         ycnt = start_yi
#         for pyi in range(start_yi,y_divided):
#             start_yi = 0
#             py = start_p[1]+step_y*pyi
#             print(xcnt,ycnt)
#             print(px,py)
#             # different angel between moveLs
#             for ang in angels:
#                 move_start = [px-side_l*cos(radians(ang/2)),py+side_l*sin(radians(ang/2)),start_p[2]]
#                 move_end = [px-side_l*cos(radians(ang/2)),py-side_l*sin(radians(ang/2)),start_p[2]]
#                 move_mid = [px,py,start_p[2]]
#                 print(move_start)
#                 print(move_end)
#                 # send_to_pose(move_start,all_Rq,conf)
#                 log = send_movels(move_start,all_Rq,conf,move_mid,all_Rq,conf,move_end,all_Rq,conf,v_profile,z10)

#                 # with open("data_param/log_"+str(v_profile.v_tcp)+"_"+"{:02d}".format(xcnt)+"_"+"{:02d}".format(ycnt)+"_"+\
#                 #             str(ang)+".csv","wb") as f:
#                 #     f.write(log)

#             ycnt +=1
#             print("Height = 0,Progress:","{:.2f}".format((xcnt*y_divided+ycnt)/(x_divided*y_divided)*100/2),'%. Total Time:',time.strftime("%H:%M:%S", time.gmtime(time.perf_counter()-st)))
#             print("=======================================")

#             pcnt += 1
#         xcnt += 1

# z_height = 100
# st = time.perf_counter()
# for vi in range(start_vi,len(all_vel)):
#     start_vi = 0
#     v_profile = all_vel[vi]

#     # from x start to end, and remeber to include the last x
#     xcnt = start_xi
#     for pxi in range(start_xi,x_divided):
#         start_xi = 0
#         px = start_p[0]+step_x*pxi
        
#         # from y start to end, and remeber to include the last x
#         ycnt = start_yi
#         for pyi in range(start_yi,y_divided):
#             start_yi = 0
#             py = start_p[1]+step_y*pyi
#             print(xcnt,ycnt)
#             print(px,py)
#             # different angel between moveLs
#             for ang in angels:
#                 move_start = [px-side_l*cos(radians(ang/2)),py+side_l*sin(radians(ang/2)),start_p[2]]
#                 move_end = [px-side_l*cos(radians(ang/2)),py-side_l*sin(radians(ang/2)),start_p[2]]
#                 move_mid = [px,py,start_p[2]+z_height]
#                 print(move_start)
#                 print(move_end)
#                 # send_to_pose(move_start,all_Rq,conf)
#                 log = send_movels(move_start,all_Rq,conf,move_mid,all_Rq,conf,move_end,all_Rq,conf,v_profile,z10)

#                 with open("data_param_zheight100/log_"+str(v_profile.v_tcp)+"_"+"{:02d}".format(xcnt)+"_"+"{:02d}".format(ycnt)+"_"+\
#                             str(ang)+".csv","wb") as f:
#                     f.write(log)
            
#             ycnt +=1
#             print("Height = 100: Progress:","{:.2f}".format((xcnt*y_divided+ycnt)/(x_divided*y_divided)*100/2),'%. Total Time:',time.strftime("%H:%M:%S", time.gmtime(time.perf_counter()-st)))
#             print("=======================================")
#         xcnt += 1
