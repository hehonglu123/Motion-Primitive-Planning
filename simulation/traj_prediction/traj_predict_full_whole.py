import numpy as np
import pathlib
from pandas import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import time

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

import sys
sys.path.append('../../toolbox')
from robots_def import *

tf.config.run_functions_eagerly(True)

robot = abb6640(R_tool=Ry(np.radians(90)),p_tool=np.array([0,0,0]))

JN = 6
h1_num = 100
h2_num = 100
Tf = 25

# model class
class DynamicNN(Model):
    def __init__(self,h1,h2,y):
        super(DynamicNN,self).__init__()
        
        self.d1 = Dense(h1, activation='relu')
        self.d2 = Dense(h2, activation='relu')
        # self.d3 = Dense(100,activation='relu')
        self.out = Dense(y)
    
    def call(self,x):
        x = self.d1(x)
        x = self.d2(x)
        # x = self.d3(x)
        return self.out(x)


model = DynamicNN(h1_num,h2_num,Tf*6)
model.load_weights('models/0400.ckpt')

def load_data(path_list,train_num,test_num):
        
    total_train_data = 0
    total_test_data = 0
    
    test_inputs = []
    test_labels = []

    for test_path in path_list[train_num:train_num+test_num]:
        data_root = pathlib.Path(test_path+'/input')
        traj_length = len(list(data_root.glob('*')))
        total_test_data += traj_length

        for traj_i in range(traj_length):
            # load input and labels
            col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
            data = read_csv(test_path+"/input/"+str(traj_i)+".csv", names=col_names)
            data_labels = read_csv(test_path+"/label/"+str(traj_i)+".csv", names=col_names)
            this_input = np.array([])
            this_label = np.array([])
            for ji in range(JN):
                curve_q=np.array(data['q'+str(ji+1)].tolist())
                curve_q_labels=np.array(data_labels['q'+str(ji+1)].tolist())
                this_input = np.append(this_input,curve_q)
                # this_input = np.append(this_input,np.deg2rad(curve_q))
                this_label = np.append(this_label,curve_q_labels-curve_q[Tf+1:])
                # print(curve_q_labels-curve_q[Tf+1:])
            test_inputs.append(this_input)
            test_labels.append(this_label)

    # print("j1 inputs len:",len(test_inputs_all[0]))
    # print("j1 labels len:",len(test_labels_all[0]))

    test_inputs = tf.convert_to_tensor(np.array(test_inputs), dtype=tf.float64)
    test_labels = tf.convert_to_tensor(np.array(test_labels), dtype=tf.float64)

    print(test_inputs.shape)
    print(test_labels.shape)

    print("Total Train Segments:",total_train_data)
    print("Total Test Segments:",total_test_data)

    return test_inputs,test_labels

# prepare data
data_path = '/media/eric/Transcend/Motion-Primitive-Planning/simulation/traj_prediction/data/data_L_z10_split'
data_root = pathlib.Path(data_path)
all_data_paths = list(data_root.glob('*'))
all_data_paths = sorted([str(path) for path in all_data_paths])
# print(all_data_paths)
train_traj_num = 121
test_traj_num = 1
test_inputs,test_labels = load_data(all_data_paths,train_traj_num,test_traj_num)

# loss object and optimizer
loss_object = tf.keras.losses.MeanSquaredError()

test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')

@tf.function
def test_step(inputs, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(inputs, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(labels, predictions)

    return predictions

the_drawing_list = []
for i in range(JN):
    the_drawing_list.append([])

input_tcp = []
predt_tcp = []
label_tcp = []

for li in range(len(test_inputs)):

    input_q = []
    predt_q = []
    label_q = []
    predictions = test_step(tf.reshape(test_inputs[li],(1,len(test_inputs[li]))), \
        tf.reshape(test_labels[li],(1,len(test_labels[li]))))
    

    for ji in range(JN):
        this_data_draw = np.vstack((np.vstack((test_labels[li].numpy()[Tf*ji:Tf*(ji+1)]+test_inputs[li].numpy()[(2*Tf+1)*ji+Tf+1:(2*Tf+1)*(ji+1)],\
                        test_inputs[li].numpy()[(2*Tf+1)*ji+Tf+1:(2*Tf+1)*(ji+1)])),
                        tf.reshape(predictions,(predictions.shape[1],)).numpy()[Tf*ji:Tf*(ji+1)]+test_inputs[li].numpy()[(2*Tf+1)*ji+Tf+1:(2*Tf+1)*(ji+1)])).T
        the_drawing_list[ji].append(this_data_draw)
        input_q.append(test_inputs[li].numpy()[(2*Tf+1)*ji+Tf+1:(2*Tf+1)*(ji+1)])
        predt_q.append(tf.reshape(predictions,(predictions.shape[1],)).numpy()[Tf*ji:Tf*(ji+1)]+test_inputs[li].numpy()[(2*Tf+1)*ji+Tf+1:(2*Tf+1)*(ji+1)])
        label_q.append(test_labels[li].numpy()[Tf*ji:Tf*(ji+1)]+test_inputs[li].numpy()[(2*Tf+1)*ji+Tf+1:(2*Tf+1)*(ji+1)])
    
    input_q = np.array(input_q).T
    predt_q = np.array(predt_q).T
    label_q = np.array(label_q).T
    
    this_input_tcp = []
    this_predt_tcp = []
    this_label_tcp = []
    for qi in range(len(input_q)):
        this_input_tcp.append(robot.fwd(np.deg2rad(input_q[qi])).p)
        this_predt_tcp.append(robot.fwd(np.deg2rad(predt_q[qi])).p)
        this_label_tcp.append(robot.fwd(np.deg2rad(label_q[qi])).p)
    input_tcp.append(np.array(this_input_tcp))
    predt_tcp.append(np.array(this_predt_tcp))
    label_tcp.append(np.array(this_label_tcp))

# prepare for visualization
# fig = plt.figure(figsize=(15,7))
# ax = fig.add_subplot(2,3)
fig, ax = plt.subplots(3,3)
fig.set_size_inches(16, 10)
# ax[0,0] = plt.axes(projection='3d')

def animate(li):
    for ji in range(JN):

        ax_r = int(ji/3)
        ax_c = int(ji%3)
        ax[ax_r,ax_c].clear()
        ax[ax_r,ax_c].plot(the_drawing_list[ji][li])
        ax[ax_r,ax_c].title.set_text('Joint '+str(ji+1))
        # draw input 
        # ax[ax_r,ax_c].plot(test_inputs_all[ji][li].numpy()[Tf+1],'red')
        # # draw prediction
        # ax[ax_r,ax_c].plot(tf.reshape(predictions,(1,Tf)).numpy(),'blue')
    # ax[0,0].clear()
    # ax[0,0].plot3D(label_tcp[li][:,0], label_tcp[li][:,1], label_tcp[li][:,2], 'blue')
    # ax[0,0].plot3D(input_tcp[li][:,0], input_tcp[li][:,1], input_tcp[li][:,2], 'orange')
    # ax[0,0].plot3D(predt_tcp[li][:,0], predt_tcp[li][:,1], predt_tcp[li][:,2], 'green')

    for pxyz_i in range(3):
        draw_pxyz = [label_tcp[li][:,pxyz_i]]
        draw_pxyz.append(input_tcp[li][:,pxyz_i])
        draw_pxyz.append(predt_tcp[li][:,pxyz_i])
        ax[2,pxyz_i].clear()
        ax[2,pxyz_i].plot(np.array(draw_pxyz).T)

        if pxyz_i == 0:
            ax[2,pxyz_i].title.set_text('Position x')
        if pxyz_i == 1:
            ax[2,pxyz_i].title.set_text('Position y')
        if pxyz_i == 2:
            ax[2,pxyz_i].title.set_text('Position z')

    fig.suptitle('Joint Angle 1~6 Step:'+str(li))
    plt.legend(['Label','Input','Predict'])
    print(li)

print(len(test_inputs))
ani = FuncAnimation(fig, animate, frames=len(test_inputs), interval=100, repeat=False)
# plt.show()

writergif = animation.PillowWriter(fps=10) 
ani.save('result_test.gif', writer=writergif)

# for li in range(len(test_inputs_all[0])):

#     for ji in range(JN):
#         predictions = test_step(test_inputs_all[li], test_labels_all[li], ji)

#         ax_r = int(ji/3)
#         ax_c = int(ji%3)+1
#         ax[ax_r,ax_c].clear()
#         ax[ax_r,ax_c].scatter([0],[0],color_check[(ji+li)%6])
    
#     time.sleep(0.1)

        