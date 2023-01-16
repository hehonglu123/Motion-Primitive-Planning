from abb_motion_program_exec import *
import numpy as np
q1=np.degrees([0.9055814868181696,-0.44826251603125206,0.6687007640627649,-2.569475617803603,1.5909190743011443,-1.4819404225038042])
q2=np.degrees([-0.07431680191557585,-0.3121309793484109,-1.9667788450226045,-2.963571949752615,1.1852471614152014,-1.0909992390040038])

mp1 = MotionProgram()
mp1.MoveAbsJ(jointtarget(q1,[0]*6),v100,fine)

mp2 = MotionProgram()
mp2.WaitTime(1)

# Execute the motion program on the robot
# Change base_url to the robot IP address
client = MotionProgramExecClient(base_url="http://192.168.55.1:80")

# Execute both motion programs simultaneously
log_results = client.execute_multimove_motion_program([mp1,mp2])
#########################################################################################
time.sleep(2)

mp1 = MotionProgram()
mp1.WaitTime(1)

mp2 = MotionProgram()
mp2.MoveAbsJ(jointtarget(q2,[0]*6),v100,fine)


# Execute the motion program on the robot
# Change base_url to the robot IP address
client = MotionProgramExecClient(base_url="http://192.168.55.1:80")

# Execute both motion programs simultaneously
log_results = client.execute_multimove_motion_program([mp1,mp2])


# client.abb_client.logout()
