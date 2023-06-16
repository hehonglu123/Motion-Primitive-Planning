from greedy import *

dataset='curve_1/'
solution_dir='curve_pose_opt2_motoman/'
data_dir='../data/'+dataset+solution_dir

###read in points
curve_js = np.loadtxt(data_dir+'Curve_js.csv',delimiter=',')

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun2.csv',\
    pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=50)

max_error_threshold=0.4
min_length=10
greedy_fit_obj=greedy_fit(robot,curve_js, min_length=min_length,max_error_threshold=max_error_threshold)

###set primitive choices, defaults are all 3
# greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy,'movec_fit':greedy_fit_obj.movec_fit_greedy}


# greedy_fit_obj.primitives={'movej_fit':greedy_fit_obj.movej_fit_greedy}
greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy}
# greedy_fit_obj.primitives={'movec_fit':greedy_fit_obj.movec_fit_greedy}

# greedy_fit_obj.primitives={'movel_fit':greedy_fit_obj.movel_fit_greedy,'movej_fit':greedy_fit_obj.movej_fit_greedy}

breakpoints,primitives,p_bp,q_bp=greedy_fit_obj.fit_under_error()
print('slope diff js (deg): ', greedy_fit_obj.get_slope_js(greedy_fit_obj.curve_fit_js,breakpoints))

############insert initial configuration#################
primitives.insert(0,'moveabsj_fit')
p_bp.insert(0,[greedy_fit_obj.curve_fit[0]])
q_bp.insert(0,[greedy_fit_obj.curve_fit_js[0]])


# breakpoints,primitives,p_bp,q_bp=greedy_fit_obj.merge_bp(breakpoints,primitives,p_bp,q_bp)
# print(breakpoints)
###plt
###3D plot
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(greedy_fit_obj.curve[:,0], greedy_fit_obj.curve[:,1],greedy_fit_obj.curve[:,2], 'gray',label='original')

ax.plot3D(greedy_fit_obj.curve_fit[:,0], greedy_fit_obj.curve_fit[:,1], greedy_fit_obj.curve_fit[:,2],'green',label='fitting')
plt.legend()
plt.show()

###adjust breakpoint index
breakpoints[1:]=breakpoints[1:]-1

print(len(breakpoints))
print(len(primitives))
print(len(p_bp))
print(len(q_bp))

df=DataFrame({'breakpoints':breakpoints,'primitives':primitives,'p_bp':p_bp,'q_bp':q_bp})
df.to_csv('greedy_output/command.csv',header=True,index=False)
df=DataFrame({'x':greedy_fit_obj.curve_fit[:,0],'y':greedy_fit_obj.curve_fit[:,1],'z':greedy_fit_obj.curve_fit[:,2],\
	'R1':greedy_fit_obj.curve_fit_R[:,0,0],'R2':greedy_fit_obj.curve_fit_R[:,0,1],'R3':greedy_fit_obj.curve_fit_R[:,0,2],\
	'R4':greedy_fit_obj.curve_fit_R[:,1,0],'R5':greedy_fit_obj.curve_fit_R[:,1,1],'R6':greedy_fit_obj.curve_fit_R[:,1,2],\
	'R7':greedy_fit_obj.curve_fit_R[:,2,0],'R8':greedy_fit_obj.curve_fit_R[:,2,1],'R9':greedy_fit_obj.curve_fit_R[:,2,2]})
df.to_csv('greedy_output/curve_fit.csv',header=True,index=False)
DataFrame(greedy_fit_obj.curve_fit_js).to_csv('greedy_output/curve_fit_js.csv',header=False,index=False)