#!/usr/bin/env python

import rpi_abb_irc5
import time
import math
import numpy as np

def main():
    try:
        egm=rpi_abb_irc5.EGM()
        t1=time.time()
        while True:
            res, state=egm.receive_from_robot(.01)
            # angle=np.deg2rad(60*((time.time()-t1)/10))
            if res:
                ###read in degrees
                print ("ID: " + str(state.robot_message.header.seqno) + " Joints: " + str(state.joint_angles))
                ###send radians
                angle_send=np.array([0,0,0,0,0,np.sin(time.time()/10.)])
                print(angle_send)
                egm.send_to_robot(angle_send)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
