import matplotlib.pyplot as plt
from general_robotics_toolbox import *
from pandas import *
import sys, traceback
import numpy as np
sys.path.append('../toolbox')
from robot_def import *

def Rx(theta):
	return np.array(([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]))
def Ry(theta):
	return np.array(([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]))
def Rz(theta):
	return np.array(([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]))



def multisplit(s, delims):
	pos = 0
	for i, c in enumerate(s):
		if c in delims:
			yield s[pos:i]
			pos = i + 1
	yield s[pos:]

def extract_points(primitive_type,points):
	if primitive_type=='movec_fit':
		endpoints=points[8:-3].split('array')
		endpoint1=endpoints[0][:-4].split(',')
		endpoint2=endpoints[1][2:].split(',')

		return list(map(float, endpoint1)),list(map(float, endpoint2))
	else:
		endpoint=points[8:-3].split(',')
		return list(map(float, endpoint))

def quadrant(q):
	temp=np.ceil(np.array([q[0],q[3],q[5]])/(np.pi/2))

	return np.hstack((temp,[0])).astype(int)
def format_point(point,quat,cf,eax):

	point_out='[['+str(point[0])+','+str(point[1])+','+str(point[2])+'],['+\
				str(quat[0])+','+str(quat[1])+','+str(quat[2])+','+str(quat[3])+'],['+\
				str(cf[0])+','+str(cf[1])+','+str(cf[2])+','+str(cf[3])+'],'+eax+']'

	return point_out

def format_movel(q,point):
	quat=R2q(fwd(q).R)
	cf=quadrant(q)


	eax='[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]'
	speed='v500'
	zone='z1'
	p=format_point(point,quat,cf,eax)
	return 'MoveL '+p+','+speed+','+zone+',Paintgun;'
def format_movej(q):

	eax='[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]'
	speed='v500'
	zone='z1'
	q_deg=np.degrees(q)
	return 'MoveAbsJ '+'[['+str(q_deg[0])+','+str(q_deg[1])+','+str(q_deg[2])+','+str(q_deg[3])+','+str(q_deg[4])+','+str(q_deg[5])+'],'+eax+'],'\
			+speed+','+zone+',Paintgun;'
def format_movec(q1,q2,point1,point2):
	quat1=R2q(fwd(q1).R)
	cf1=quadrant(q1)
	quat2=R2q(fwd(q2).R)
	cf2=quadrant(q2)

	eax='[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]'
	speed='v500'
	zone='z1'

	p1=format_point(point1,quat1,cf1,eax)
	p2=format_point(point2,quat2,cf2,eax)
	return 'MoveC '+p1+','+p2+','+speed+','+zone+',Paintgun;'

def main():

	col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
	data = read_csv("../data/from_cad/Curve_backproj_js.csv", names=col_names)
	curve_q1=data['q1'].tolist()
	curve_q2=data['q2'].tolist()
	curve_q3=data['q3'].tolist()
	curve_q4=data['q4'].tolist()
	curve_q5=data['q5'].tolist()
	curve_q6=data['q6'].tolist()
	curve_js=np.vstack((curve_q1, curve_q2, curve_q3,curve_q4,curve_q5,curve_q6)).T

	data = read_csv("command_backproj.csv")
	breakpoints=data['breakpoints'].tolist()
	primitives=data['primitives'].tolist()
	points=data['points'].tolist()

	commands=[]
	for i in range(len(breakpoints)):
		if primitives[i]=='movel_fit':
			point=extract_points(primitives[i],points[i])
			command_temp=format_movel(curve_js[breakpoints[i]],point)

		elif primitives[i]=='movec_fit':
			point1,point2=extract_points(primitives[i],points[i])
			command_temp=format_movec(curve_js[breakpoints[i-1]],curve_js[breakpoints[i]],point1,point2)
		else:
			point=extract_points(primitives[i],points[i])
			command_temp=format_movej(point)
		print(command_temp)

if __name__ == "__main__":
	main()