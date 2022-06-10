from general_robotics_toolbox import *
sys.path.append('../../../toolbox')
from robots_def import *

robot = m900ia(d=50)

q=[0.3254,0.1459,-0.6167,-0.6065,0.5724,0.4408]

J = robot.jacobian(q)

print(J)

