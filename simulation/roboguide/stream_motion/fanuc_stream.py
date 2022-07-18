import socket
import select
import errno
from copy import deepcopy
from struct import *
import time
from tkinter.messagebox import showerror
import numpy as np
from numpy import deg2rad,rad2deg

def fanuc2usual(q):
    q[2]=q[1]+q[2] #j3_usual=j2_fanuc+j3_fanuc
    return q
def usual2fanuc(q):
    q[2]=q[2]-q[1] #j3_fanuc=j3_usual-j2_fanuc
    return q

# get binary function
getbinary = lambda x, n: format(x, 'b').zfill(n)

class MotionStream(object):
    def __init__(self,udp_ip='127.0.0.2',udp_port=60015) -> None:
        
        # UDP socket
        self.robot_ip=udp_ip
        self.serverAddr=(udp_ip,udp_port)
        self.sock=socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.sock.settimeout(10.0)

        # physical (or simulation) robot
        self.robot_ready=False
        self.robot_joint=None
        self.robot_ts=None
        self.robot_seq=None
        self.send_seq=None
        self.ver_num=None

    def start_stream(self):

        # initial packet
        init_pkt = pack('>2I',0,1)
        self.sock.sendto(init_pkt, self.serverAddr)
    
    def end_stream(self):

        # stop packet
        end_pkt = pack('>2I',2,1)
        self.sock.sendto(end_pkt, self.serverAddr)
    
    def receive_from_robot(self,timeout=0):

        s=self.sock
        s_list=[s]
        try:
            res=select.select(s_list, [], s_list, timeout)
        except select.error as err:
            if err.args[0] == errno.EINTR:
                return False, None
            else:
                raise
        
        if len(res[0]) == 0 and len(res[2])==0:
            return False, None
        try:
            # (buf, addr)=s.recvfrom(65536)
            (data, addr)=s.recvfrom(132) # a status package is 132 byte
        except:
            return False, None
        
        data_unpack=unpack('>3I2B3HI27f',data)

        # get the sequence number
        self.robot_seq=data_unpack[2]
        # if not self.robot_ready:
        #     self.send_seq=data_unpack[2]
        self.send_seq=data_unpack[2]
        # get version number
        self.ver_num=data_unpack[1]

        # robot status
        status_bit=getbinary(data_unpack[3],4)
        if status_bit[3] == '1' and status_bit[1] == '1':
            # the robot is ready to move
            self.robot_ready=True
        
        if status_bit[3] == '0':
            # the robot is stopped
            self.robot_ready=False
        
        # robot info
        self.robot_joint=fanuc2usual(np.deg2rad(data_unpack[18:24]))
        self.robot_ts=data_unpack[8]

        return True,(self.robot_seq,self.robot_ts,self.robot_joint)
    
    def send_joint(self,joint_angles,last_data=0):

        cmd_q=usual2fanuc(rad2deg(joint_angles))
        # send motion package
        cmd_pkt = pack('>3I2B2H2B4H9f',1,self.ver_num,self.send_seq,last_data,\
            0,0,0,1,0,0,0,0,0,\
            cmd_q[0],cmd_q[1],cmd_q[2],cmd_q[3],cmd_q[4],cmd_q[5],0,0,0)
        
        try:
            self.sock.sendto(cmd_pkt, self.serverAddr)
        except:
            return False
        # sequence number +1 after send a command
        # self.send_seq += 1
        return True
    
    def clear_queue(self):
        # # Clear UDP queue
        while True:
            res_i, state_i = self.receive_from_robot()
            if not res_i:
                break