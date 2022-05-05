from shutil import move
import numpy as np
from fanuc_client import *
import general_robotics_toolbox as rox
from matplotlib import pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

client = FANUCClient(robot_ip='127.0.0.2')

qddot_nominal = [500,800,800,1000,1000,1000]
q_target = [45,60,60,180,90,180]

for ji in range(6):
    jt1 = jointtarget(1,1,2,np.zeros(6),[0]*6)
    joint_target = np.zeros(6)
    joint_target[ji] = q_target[ji]
    jt2 = jointtarget(1,1,2,joint_target,[0]*6)

    tp = TPMotionProgram()
    tp.moveJ(jt2,100,'%',-1)
    tp.moveJ(jt1,100,'%',-1)

    # print(tp.get_tp())

    res = client.execute_motion_program(tp)

    res_string = res.decode('utf-8')

    log_stamp = np.array([])
    log_q = np.array([])
    for row in res_string.split('\n')[1:]:
        if row == '':
            continue
        data = row.split(', ')
        log_stamp = np.append(log_stamp,float(data[0])/1000)
        log_q = np.append(log_q,float(data[ji+1]))
    
    q = np.array([])
    stamp=np.array([])
    for i in range(len(log_q)):
        this_q = log_q[i]
        this_stamp = log_stamp[i]
        if i>5 and i<len(log_q)-5:
            # if the recording is not fast enough
            # then having to same logged joint angle
            # do interpolation for estimation
            if np.all(this_q==log_q[i+1]):
                last_cont = True
                continue
        q = np.append(q,this_q)
        stamp = np.append(stamp,this_stamp)
        last_cont = False
    
    qdot = np.divide(np.diff(q),np.diff(stamp))
    qddot = np.divide(np.diff(qdot),np.diff(stamp)[:-1])
    qddot_f = np.delete(qddot,np.argwhere(qddot>qddot_nominal[ji]))
    qddot_f = np.delete(qddot_f,np.argwhere(qddot_f<-qddot_nominal[ji]))
    # print(qdot)
    print(qddot)
    # qddot_f = moving_average(qddot,3)
    # plt.plot(qddot_f)
    plt.plot(qddot)
    plt.show()
