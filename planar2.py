import matplotlib.pyplot as plt
import numpy as np
import pwlf
import copy, time
from scipy.stats import linregress

# Similation parameters
dt = 0.01

# Link lengths
l1 = l2 = 1




def plot_arm(theta1, theta2, target_x, target_y):  # pragma: no cover
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + \
        np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    plt.cla()

    plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
    plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

    plt.plot(shoulder[0], shoulder[1], 'ro')
    plt.plot(elbow[0], elbow[1], 'ro')
    plt.plot(wrist[0], wrist[1], 'ro')

    plt.plot([wrist[0], target_x], [wrist[1], target_y], 'g--')
    plt.plot(target_x, target_y, 'g*')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.show()
    plt.pause(dt)

def forward(theta1,theta2):
    return np.cos(theta1)+np.cos(theta1+theta2),np.sin(theta1)+np.sin(theta1+theta2)

def inv(x,y):
    theta2=np.arccos((x**2+y**2-2)/2)
    theta1=np.arctan2(y,x)-np.arctan2(np.sin(theta2),1+np.cos(theta2))
    return theta1,theta2
    
def moveL(theta1,theta2,x,y,v=0.05):
    act_x,act_y=forward(theta1,theta2)
    total_step=int(np.linalg.norm(np.array([act_x-x,act_y-y]))/v)

    for i in range(total_step):
        theta_1,theta_2=inv(act_x+(x-act_x)*(i+1)/total_step,act_y+(y-act_y)*(i+1)/total_step)
        plot_arm(theta_1, theta_2, x, y)

    try:
        return theta_1,theta_2
    except:
        return theta1,theta2
def moveJ(theta1_start,theta2_start,theta1,theta2,v=0.05):
    x,y=forward(theta1,theta2)

    total_step=int(np.linalg.norm(np.array([theta1_start-theta1,theta2_start-theta2]))/v)
    for i in range(total_step):
        plot_arm(theta1_start+(theta1-theta1_start)*(i+1)/total_step, theta2_start+(theta2-theta2_start)*(i+1)/total_step, x, y)
    return theta1,theta2

def find_break_point(xHat,yHat):
    break_points=[]
    diff=(yHat[1]-yHat[0])/(xHat[1]-xHat[0])
    for i in range(1,len(xHat)):
        if np.abs((yHat[i]-yHat[i-1])/(xHat[i]-xHat[i-1])-diff)>0.01:
            break_points.append(xHat[i-1])
            diff=(yHat[i]-yHat[i-1])/(xHat[i]-xHat[i-1])
    break_points.append(xHat[-1])
    return break_points

def fit_seg(my_pwlf,seg_num):
    res = my_pwlf.fit(seg_num)

    # predict for the determined points
    xHat = np.linspace(min(my_pwlf.x_data), max(my_pwlf.x_data), num=10000)
    yHat = my_pwlf.predict(xHat)

    return xHat,yHat, np.max(np.abs(yHat-np.sin(xHat*np.pi/1.5+0.1)))

def fit_break(my_pwlf,break_points):
    ###known break points
    res = my_pwlf.fit_with_breaks(break_points)
    # predict for the determined points
    xHat = np.linspace(min(my_pwlf.x_data), max(my_pwlf.x_data), num=10000)
    yHat = my_pwlf.predict(xHat)

    return xHat,yHat, np.max(np.abs(yHat-np.sin(xHat*np.pi/1.5+0.1)))

def break_slope(x,y,threshold=0.5,step_size=5):
    
    break_points=[0]
    break_point_idx=0
    res=linregress(x[:10], y[:10])
    prev_slope=res.slope

    for i in range(10,len(x),step_size):
        res=linregress(x[break_point_idx:i], y[break_point_idx:i])
        prev_slope=res.slope

        res=linregress(x[i:i+step_size], y[i:i+step_size])
        if abs(res.slope-prev_slope)>threshold:
            break_points.append(i)
            break_point_idx=i 
    break_points.append(-1)
    return break_points

def main():
    plt.ion()

    error_threshold=0.08
    max_segments=10
    num_seg=int(max_segments/2.)
    seg_up=copy.deepcopy(max_segments)
    seg_down=1
    cur_error=1.
    min_diff=99

    theta1,theta2=0,0
    fig = plt.figure()

    ###pwlf settins
    x=np.arange(-1.5,1.5,0.001)
    y=np.sin(x*np.pi/1.5+0.1)

    # initialize piecewise linear fit with your x and y train_data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    print(num_seg)
    xHat,yHat,cur_error=fit_seg(my_pwlf,num_seg)

    search_dict={num_seg:[xHat,yHat,cur_error]}
    while seg_up-seg_down>1:
        

        if cur_error>error_threshold:
            seg_down=copy.deepcopy(num_seg)
            num_seg=round((num_seg+seg_up)/2.)
            print(num_seg)
            try:
                xHat,yHat,cur_error=search_dict[num_seg][0],search_dict[num_seg][1],search_dict[num_seg][2]
            except:
                xHat,yHat,cur_error=fit_seg(my_pwlf,num_seg)
                search_dict[num_seg]=[xHat,yHat,cur_error]
        else:
            seg_up=copy.deepcopy(num_seg)
            num_seg=round((num_seg+seg_down)/2.)
            print(num_seg)
            try:
                xHat,yHat,cur_error=search_dict[num_seg][0],search_dict[num_seg][1],search_dict[num_seg][2]
            except:
                xHat,yHat,cur_error=fit_seg(my_pwlf,num_seg)
                search_dict[num_seg]=[xHat,yHat,cur_error]

    if cur_error>error_threshold:
        try:
            xHat,yHat,cur_error=search_dict[num_seg+1][0],search_dict[num_seg+1][1],search_dict[num_seg+1][2]
        except:
            xHat,yHat,cur_error=fit_seg(my_pwlf,num_seg+1)
            search_dict[num_seg+1]=[xHat,yHat,cur_error]
    x0=find_break_point(xHat,yHat)

    theta1_first,theta2_first=inv(xHat[0],yHat[0])
    #moveJ to start position
    theta1,theta2=moveJ(theta1,theta2,theta1_first,theta2_first)

    for xxx in x0:
        theta1,theta2=moveL(theta1,theta2,xxx,yHat[min(range(len(xHat)), key=lambda i: abs(xHat[i]-xxx))])

    print('num segments: ',num_seg)
    print('max error: ',cur_error)

def main2():
    x=np.arange(-1.5,1.5,0.001)
    y=np.sin(x*np.pi/1.5+0.1)
    
    # initialize piecewise linear fit with your x and y train_data
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    found=False
    error_threshold=0.08
    slope_threshold_prev=0.
    slope_threshold=1.
    break_points=break_slope(x,y,slope_threshold)
    num_seg=100
    num_seg_prev=len(break_points)

    xHat,yHat,cur_error=fit_break(my_pwlf,break_points)

    search_dict={slope_threshold:[xHat,yHat,[0,2*slope_threshold],cur_error]}

    while abs(num_seg_prev-num_seg)>1 or not found:

        if cur_error>error_threshold:
            slope_threshold_prev=copy.deepcopy(slope_threshold)
            xHat,yHat,search_range,cur_error=search_dict[slope_threshold]
            slope_threshold=(slope_threshold+search_range[0])/2.
            try:
                xHat,yHat,search_range,cur_error=search_dict[slope_threshold]
            except:
                break_points=break_slope(x,y,slope_threshold)
                print(len(break_points),cur_error)
                num_seg_prev=copy.deepcopy(num_seg)
                num_seg=len(break_points)

                xHat,yHat,cur_error=fit_break(my_pwlf,break_points)
                search_dict[slope_threshold]=[xHat,yHat,[search_range[0],slope_threshold_prev],cur_error]
        else:
            found=True
            xHat,yHat,search_range,cur_error=search_dict[slope_threshold]
            slope_threshold=(slope_threshold+search_range[-1])/2.

            try:
                xHat,yHat,search_range,cur_error=search_dict[slope_threshold]
            except:
                break_points=break_slope(x,y,slope_threshold)
                num_seg_prev=copy.deepcopy(num_seg)
                num_seg=len(break_points)
                xHat,yHat,cur_error=fit_break(my_pwlf,break_points)
                search_dict[slope_threshold]=[xHat,yHat,[slope_threshold_prev,search_range[-1]],cur_error]


    if cur_error>error_threshold:
        xHat,yHat,search_range,cur_error=search_dict[search_range[0]]

    print('num segments: ',num_seg)
    print('max error: ',cur_error)



if __name__ == "__main__":
    main2()
