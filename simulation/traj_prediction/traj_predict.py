import numpy as np
import pathlib
from pandas import *

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

tf.config.run_functions_eagerly(True)

JN = 6
h1_num = 100
h2_num = 100
Tf = 25

# hyper parameters
batch_size = 256
lr = 0.001
EPOCHS = 100

# model class
class DynamicNN(Model):
    def __init__(self,h1,h2,y):
        super(DynamicNN,self).__init__()
        
        self.d1 = Dense(h1, activation='relu')
        self.d2 = Dense(h2, activation='relu')
        self.d3 = Dense(y)
    
    def call(self,x):
        # x = self.d1(x)
        # x = self.d2(x)
        return self.d3(self.d2(self.d1(x)))


models = []
for i in range(JN):
    models.append(DynamicNN(h1_num,h2_num,Tf))

def load_data(path_list,train_num,test_num):
    # number_samples = len(path_list)
    # Images = []
    # for each_path in path_list:
    #     img = plt.imread(each_path)
    #     # divided by 255.0
    #     img = img.reshape(784, 1) / 255.0
    #     '''
    #     In some cases data need to be preprocessed by subtracting the mean value of the data and divided by the 
    #     standard deviation to make the data follow the normal distribution.
    #     In this assignment, there will be no penalty if you don't do the process above.
    #     '''
    #     # DONT add bias
    #     # img = np.vstack((img, [1]))
    #     Images.append(img)
    # data = tf.convert_to_tensor(np.array(Images).reshape(number_samples, 784), dtype=tf.float32)
    
    total_train_data = 0
    total_test_data = 0

    train_inputs_all = []
    train_labels_all = []
    for ji in range(JN):
        train_inputs_all.append([])
    for ji in range(JN):
        train_labels_all.append([])

    for train_path in path_list[:train_num]:
        data_root = pathlib.Path(train_path+'/input')
        traj_length = len(list(data_root.glob('*')))
        total_train_data += traj_length

        for traj_i in range(traj_length):
            # load input
            col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
            data = read_csv(train_path+"/input/"+str(traj_i)+".csv", names=col_names)
            for ji in range(JN):
                curve_q=np.array(data['q'+str(ji+1)].tolist())
                train_inputs_all[ji].append(curve_q)
            # load label
            col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
            data = read_csv(train_path+"/label/"+str(traj_i)+".csv", names=col_names)
            for ji in range(JN):
                curve_q=np.array(data['q'+str(ji+1)].tolist())
                train_labels_all[ji].append(curve_q)
    
    test_inputs_all = []
    test_labels_all = []
    for ji in range(JN):
        test_inputs_all.append([])
    for ji in range(JN):
        test_labels_all.append([])

    for test_path in path_list[train_num:train_num+test_num]:
        data_root = pathlib.Path(test_path+'/input')
        traj_length = len(list(data_root.glob('*')))
        total_test_data += traj_length

        for traj_i in range(traj_length):
            # load input
            col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
            data = read_csv(test_path+"/input/"+str(traj_i)+".csv", names=col_names)
            for ji in range(JN):
                curve_q=np.array(data['q'+str(ji+1)].tolist())
                test_inputs_all[ji].append(curve_q)
            # load label
            col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
            data = read_csv(test_path+"/label/"+str(traj_i)+".csv", names=col_names)
            for ji in range(JN):
                curve_q=np.array(data['q'+str(ji+1)].tolist())
                test_labels_all[ji].append(curve_q)
    # print("j1 inputs len:",len(test_inputs_all[0]))
    # print("j1 labels len:",len(test_labels_all[0]))

    for ji in range(JN):
        train_inputs_all[ji] = tf.convert_to_tensor(np.array(train_inputs_all[ji]), dtype=tf.float32)
        train_labels_all[ji] = tf.convert_to_tensor(np.array(train_labels_all[ji]), dtype=tf.float32)
        test_inputs_all[ji] = tf.convert_to_tensor(np.array(test_inputs_all[ji]), dtype=tf.float32)
        test_labels_all[ji] = tf.convert_to_tensor(np.array(test_labels_all[ji]), dtype=tf.float32)
    # print(train_inputs_all[0][0])

    print("Total Train Segments:",total_train_data)
    print("Total Test Segments:",total_test_data)

    return train_inputs_all,train_labels_all,test_inputs_all,test_labels_all

# prepare data
data_path = '/media/eric/Transcend/Motion-Primitive-Planning/simulation/traj_prediction/data/data_L_z10_split'
data_root = pathlib.Path(data_path)
all_data_paths = list(data_root.glob('*'))
all_data_paths = sorted([str(path) for path in all_data_paths])
# print(all_data_paths)
train_traj_num = 120
test_traj_num = 30
train_inputs_all,train_labels_all,test_inputs_all,test_labels_all = load_data(all_data_paths,train_traj_num,test_traj_num)

train_ds_all = []
test_ds_all = []
for ji in range(JN):
    train_ds_all.append(tf.data.Dataset.from_tensor_slices((train_inputs_all[ji], train_labels_all[ji])).shuffle(10000).batch(batch_size))
    test_ds_all.append(tf.data.Dataset.from_tensor_slices((test_inputs_all[ji], test_labels_all[ji])).shuffle(10000).batch(batch_size))

print(train_ds_all[0])

# loss object and optimizer
loss_objects = []
for i in range(JN):
    loss_objects.append(tf.keras.losses.MeanSquaredError())
optimizers = []
for i in range(JN):
    optimizers.append(tf.keras.optimizers.Adam(learning_rate=lr))

train_loss_all = []
for i in range(JN):
    train_loss_all.append(tf.keras.metrics.MeanSquaredError(name='train_loss_'+str(i)))
test_loss_all = []
for i in range(JN):
    test_loss_all.append(tf.keras.metrics.MeanSquaredError(name='test_loss_'+str(i)))

# train function
@tf.function
def train_step(inputs, labels, jn_i):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = models[jn_i](inputs, training=True)
        loss = loss_objects[jn_i](labels, predictions)
    gradients = tape.gradient(loss, models[jn_i].trainable_variables)
    optimizers[jn_i].apply_gradients(zip(gradients, models[jn_i].trainable_variables))

    # print(loss)
    train_loss_all[jn_i](labels, predictions)

@tf.function
def test_step(inputs, labels, jn_i):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = models[jn_i](inputs, training=False)
    t_loss = loss_objects[jn_i](labels, predictions)

    test_loss_all[jn_i](labels, predictions)

train_loss_save = []
for i in range(JN):
    train_loss_save.append([])
test_loss_save = []
for i in range(JN):
    test_loss_save.append([])

checkpoint_path = "model/{epoch:04d}_j{joint:01d}.ckpt"
save_weight_every = 4
for epoch in range(EPOCHS):

    print("Epoch:",epoch+1)
    for ji in range(JN):

        # Reset the metrics at the start of the next epoch
        train_loss_all[ji].reset_states()
        test_loss_all[ji].reset_states()

        for inputs, labels in train_ds_all[ji]:
            train_step(inputs, labels, ji)

        for test_inputs, test_labels in test_ds_all[ji]:
            test_step(test_inputs, test_labels, ji)

        print(
        f'Joint {ji + 1}, '
        f'Loss: {train_loss_all[ji].result()}, '
        f'Test Loss: {test_loss_all[ji].result()}, '
        )
        print("---------------------")
        train_loss_save[ji].append(train_loss_all[ji].result().numpy())
        test_loss_save[ji].append(test_loss_all[ji].result().numpy())

        if epoch % save_weight_every == 0:
            models[ji].save_weights(checkpoint_path.format(epoch=epoch,joint=ji+1))
    print("======================================")

    with open('train_loss_save.npy','wb') as f:
        np.save(f,np.array(train_loss_save))
    with open('test_loss_save.npy','wb') as f:
        np.save(f,np.array(test_loss_save))