import abb_motion_program_exec_client as abb
from threading import Thread

from io import StringIO


j1 = abb.jointtarget([10,20,30,40,50,60],[0]*6)
j2 = abb.jointtarget([90,-60,60,-93,94,-95],[0]*6)
j3 = abb.jointtarget([-80,81,-82,83,-84,85],[0]*6)

my_tool = abb.tooldata(True,abb.pose([0,0,0.1],[1,0,0,0]),abb.loaddata(0.001,[0,0,0.001],[1,0,0,0],0,0,0)) 

mp = abb.MotionProgram(tool=my_tool)
mp.MoveAbsJ(j1,abb.vmax,abb.fine)
mp.MoveAbsJ(j2,abb.vmax,abb.fine)
mp.MoveAbsJ(j3,abb.vmax,abb.fine)

# Execute the motion program on the robot
# Change base_url to the robot IP address
client1 = abb.MotionProgramExecClient(base_url="http://192.168.68.68:80")
client2 = abb.MotionProgramExecClient()

log_results1=None
log_results2=None
def move_robot1():
    global log_results1, df1, mp
    log_results1 = client1.execute_motion_program(mp)

def move_robot2():
    global log_results2, df2, mp
    log_results2 = client2.execute_motion_program(mp)

def main():
    global log_results1, log_results2, client1, client2

    ###move to start
    mp = abb.MotionProgram(tool=my_tool)
    mp.MoveAbsJ(j1,abb.v1000,abb.fine)
    client1.execute_motion_program(mp)
    client2.execute_motion_program(mp)

    t1 = Thread(target=move_robot1)
    t2 = Thread(target=move_robot2)

    # start the threads
    t1.start()
    t2.start()

    # wait for the threads to complete
    t1.join()
    t2.join()

    with open("recorded_data/log1_5.csv","wb") as f:
        f.write(log_results1)

    with open("recorded_data/log2_5.csv","wb") as f:
        f.write(log_results2)

if __name__ == '__main__':
    main()