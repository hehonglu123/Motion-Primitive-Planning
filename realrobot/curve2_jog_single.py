from abb_motion_program_exec import *
import numpy as np
q1=np.degrees([-0.8419053648429755,0.6140120331981018,0.2305977047263266,-2.7062215427861376,-0.7458494852368226,-2.2157714145005687])
q2=[82.63,-88.42,-98.89,-65.49,61.24,53.27]

mp1 = MotionProgram()
mp1.WaitTime(1)

mp2 = MotionProgram()
mp2.MoveAbsJ(jointtarget(q2,[0]*6),v100,fine)


# Execute the motion program on the robot
# Change base_url to the robot IP address
client = MotionProgramExecClient(base_url="http://192.168.55.1:80")

# Execute both motion programs simultaneously
log_results = client.execute_multimove_motion_program([mp1,mp2])

time.sleep(2)

mp1 = MotionProgram()
mp1.MoveAbsJ(jointtarget(q1,[0]*6),v100,fine)

mp2 = MotionProgram()
mp2.WaitTime(1)

# Execute the motion program on the robot
# Change base_url to the robot IP address
client = MotionProgramExecClient(base_url="http://192.168.55.1:80")

# Execute both motion programs simultaneously
log_results = client.execute_multimove_motion_program([mp1,mp2])

client.abb_client.logout()
