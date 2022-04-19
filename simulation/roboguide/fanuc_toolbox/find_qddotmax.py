from shutil import move
import numpy as np
from fanuc_client import *
import general_robotics_toolbox as rox
from matplotlib import pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

client = FANUCClient()

qddot_nominal = [500,800,800,1000,1000,1000]
q_target = [45,60,60,180,90,180]

for ji in range(6):
    jt1 = jointtarget(1,0,1,np.zeros(6),[0]*6)
    joint_target = np.zeros(6)
    joint_target[ji] = q_target[ji]
    jt2 = jointtarget(1,0,1,joint_target,[0]*6)

    tp = TPMotionProgram()
    tp.moveJ(jt2,100,'%',-1)
    tp.moveJ(jt1,100,'%',-1)
    res = client.execute_motion_program(tp)

    res_string = res.decode('utf-8')

    stamp = np.array([])
    q = np.array([])
    for row in res_string.split('\n')[1:]:
        if row == '':
            continue
        data = row.split(', ')
        stamp = np.append(stamp,float(data[0])/1000)
        q = np.append(q,float(data[ji+1]))
    
    qdot = np.divide(np.diff(q),np.diff(stamp))
    qddot = np.divide(np.diff(qdot),np.diff(stamp)[:-1])
    qddot_f = np.delete(qddot,np.argwhere(qddot>qddot_nominal[ji]))
    qddot_f = np.delete(qddot_f,np.argwhere(qddot_f<-qddot_nominal[ji]))
    # print(qdot)
    print(qddot)
    # qddot_f = moving_average(qddot,3)
    plt.plot(qddot_f)
    plt.show()
