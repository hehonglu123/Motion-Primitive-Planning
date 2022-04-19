########
# The module automatically connect to fanuc robot 
# and execute LS program
########

import numpy as np
from typing import NamedTuple
import itertools
from ftplib import FTP
from urllib.request import urlopen   
import os

class pose(NamedTuple):
    trans: np.ndarray # [x,y,z]
    rot: np.ndarray # [w,p,r] deg

class confdata(NamedTuple):
    J5: str # F or N
    J3: str # U or D
    J1: str # T or B
    turn4: int # -1 0 1
    turn5: int # -1 0 1
    turn6: int # -1 0 1

class robtarget(NamedTuple):
    group: int
    uframe: int
    utool: int
    trans: np.ndarray # [x,y,z]
    rot: np.ndarray # [w,p,r] deg
    robconf: confdata # 
    extax: np.ndarray # shape=(6,)

class jointtarget(NamedTuple):
    group: int
    uframe: int
    utool: int
    robax: np.ndarray # shape=(6,)
    extax: np.ndarray # shape=(6,)


class TPMotionProgram(object):
    def __init__(self) -> None:
        
        self.progs = []
        self.target = []
        self.t_num = 0

    def moveJ(self,target,vel,vel_unit,zone):
        '''
        
        '''
        
        mo = 'J '

        self.target.append(target)
        self.t_num += 1
        mo += 'P['+str(self.t_num)+'] '
        
        if vel_unit == 'sec':
            vel = np.min([np.max([vel,1]),32000])
            mo += str(vel) + 'sec '
        else:
            vel = np.min([np.max([vel,1]),100])
            mo += str(vel) + '% '
        
        if zone < 0:
            mo += 'FINE '
        else:
            zone = np.min([zone,100])
            mo += 'CNT'+str(zone)+' '
        
        mo += ';'

        self.progs.append(mo)

    def moveL(self,target,vel,vel_unit,zone):
        '''
        
        '''
        
        mo = 'L '

        self.target.append(target)
        self.t_num += 1
        mo += 'P['+str(self.t_num)+'] '
        
        # only support mm/sec for now
        vel = np.min([np.max([vel,1]),2000])
        mo += str(vel) + 'mm/sec '
        
        if zone < 0:
            mo += 'FINE '
        else:
            zone = np.min([zone,100])
            mo += 'CNT'+str(zone)+' '
        
        mo += ';'

        self.progs.append(mo)

    def moveC(self,mid_target,end_target,vel,vel_unit,zone):
        '''
        
        '''
        
        mo = 'C '

        # mid point
        self.target.append(mid_target)
        self.t_num += 1
        mo += 'P['+str(self.t_num)+'] \n    :  '
        # end point
        self.target.append(end_target)
        self.t_num += 1
        mo += 'P['+str(self.t_num)+'] '
        
        # only support mm/sec for now
        vel = np.min([np.max([vel,1]),2000])
        mo += str(vel) + 'mm/sec '
        
        if zone < 0:
            mo += 'FINE '
        else:
            zone = np.min([zone,100])
            mo += 'CNT'+str(zone)+' '
        
        mo += ';'

        self.progs.append(mo)

    def dump_program(self,filename):
        
        # program name, attribute, motion
        mo = '/PROG  '+filename+'\n/ATTR\n/MN\n'
        mo += '   1:  UFRAME_NUM=0 ;\n   2:  UTOOL_NUM=1 ;\n   3:  DO[101]=ON ;\n   4:  RUN DATARECORDER ;\n'
        line_num=5
        for prog in self.progs:
            mo += '   '+str(line_num)+':'
            mo += prog
            mo += '\n'
            line_num += 1
        mo += '   '+str(line_num)+':  DO[101]=OFF ;\n'

        # pose data
        mo += '/POS\n'
        for (t_num, target) in itertools.zip_longest(range(self.t_num), self.target):
            
            if type(target) == jointtarget:
                mo+='P['+str(t_num+1)+']{\n'
                mo+='   GP'+str(target.group)+':\n'
                mo+='   UF : '+str(target.uframe)+', UT : '+str(target.utool)+',\n'
                mo+='   J1 = '+format(round(target.robax[0],3),'.3f')+' deg,  J2 = '+format(round(target.robax[1],3),'.3f')+' deg,  J3 = '+format(round(target.robax[2],3)-round(target.robax[1],3),'.3f')+' deg,\n'
                mo+='   J4 = '+format(round(target.robax[3],3),'.3f')+' deg,  J5 = '+format(round(target.robax[4],3),'.3f')+' deg,  J6 = '+format(round(target.robax[5],3),'.3f')+' deg\n'
                mo+='};\n'
            if type(target) == robtarget:
                mo+='P['+str(t_num+1)+']{\n'
                mo+='   GP'+str(target.group)+':\n'
                mo+='   UF : '+str(target.uframe)+', UT : '+str(target.utool)+',     CONFIG : \''+\
                    target.robconf.J5+' '+target.robconf.J3+' '+target.robconf.J1+', '+\
                    str(target.robconf.turn4)+', '+str(target.robconf.turn5)+', '+str(target.robconf.turn6)+'\',\n'
                mo+='   X = '+format(round(target.trans[0],3),'.3f')+' mm,  Y = '+format(round(target.trans[1],3),'.3f')+' mm,  Z = '+format(round(target.trans[2],3),'.3f')+' mm,\n'
                mo+='   W = '+format(round(target.rot[0],3),'.3f')+' deg,  P = '+format(round(target.rot[1],3),'.3f')+' deg,  R = '+format(round(target.rot[2],3),'.3f')+' deg\n'
                mo+='};\n'
        
        # end
        mo += '/END\n'

        with open(filename+'.LS', "w") as f:
            f.write(mo)

class FANUCClient(object):
    def __init__(self,robot_ip='127.0.0.2') -> None:

        self.robot_ip = robot_ip
        self.robot_ftp = FTP(self.robot_ip,user='robot')
        self.robot_ftp.login()
        self.robot_ftp.cwd('UD1:')
    
    def execute_motion_program(self, tpmp: TPMotionProgram):

        # # save a temp
        tpmp.dump_program('TMP')

        # # copy to robot via ftp
        with open('TMP.LS','rb') as the_prog:
            self.robot_ftp.storlines('STOR TMP.LS',the_prog)

        motion_url='http://'+self.robot_ip+'/karel/remote'
        res = urlopen(motion_url)

        file_url='http://'+self.robot_ip+'/ud1/log.txt'
        res = urlopen(file_url)

        if os.path.exists("TMP.LS"):
            os.remove("TMP.LS")
        else:
            print("TMP.LS is deleted.")

        return res.read()

def main():

    tp = TPMotionProgram()

    pt1 = robtarget(1,0,1,[1850,200,290],[-180,0,0],confdata('N','U','T',0,0,0),[0]*6)
    pt2 = robtarget(1,0,1,[1850,200,589],[-180,0,0],confdata('N','U','T',0,0,0),[0]*6)
    jt1 = jointtarget(1,0,1,[0,20,-10,0,-20,10],[0]*6)
    pt3 = robtarget(1,0,1,[1850,250,400],[-180,0,0],confdata('N','U','T',0,0,0),[0]*6)
    
    tp.moveL(pt1,50,'mmsec',100)
    tp.moveL(pt2,50,'mmsec',-1)
    tp.moveJ(jt1,100,'%',-1)
    tp.moveL(pt2,50,'mmsec',100)
    tp.moveC(pt3,pt1,50,'mmsec',-1)
    tp.moveL(pt2,50,'mmsec',-1)

    client = FANUCClient()
    res = client.execute_motion_program(tp)

    with open("fanuc_log.csv","wb") as f:
        f.write(res)
    
    print(res.decode('utf-8'))

    tp = TPMotionProgram()
    tp.moveL(pt1,50,'mmsec',100)
    tp.moveL(pt2,50,'mmsec',-1)
    client = FANUCClient()
    res = client.execute_motion_program(tp)

if __name__ == "__main__":
    main()
