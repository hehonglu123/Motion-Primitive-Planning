from lambda_calc import *


qdot_prev=np.array([0.39470602, 	-0.82688407,  	0.62662043,  	0.16406161,  	1.37786894,  	2.47599701])
qdot_d=np.array([	0.40274907, 	-0.80726031,  	0.6113778,   	0.14283573,  	1.37054176,  	2.52335554])

joint_acc_limit=[	5.44542727,  	5.09636142,  	7.29547627, 	42.0100751,  	27.00024353, 	59.34119457]
qddot_d=np.array([	2.09295457,  	5.10647209, 	-3.96642176, 	-5.52337493, 	-1.90667086, 	12.32358297])

dt=0.003842918363703802

print((0.99*qdot_d-qdot_prev)/dt)
# alpha=fminbound(q_linesearch,0,1,args=(qdot_prev,qdot_d,dt,joint_acc_limit))
# print(alpha)
