import numpy as np
import pathlib
from pandas import *

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

# tf.config.run_functions_eagerly(True)

JN = 6
h1_num = 100
h2_num = 100
Tf = 25

# hyper parameters
batch_size = 256
lr = 0.1
EPOCHS = 1000

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

def load_data(path_list,train_num,test_num):
        
    total_train_data = 0
    total_test_data = 0

    train_inputs = []
    train_labels = []

    for train_path in path_list[:train_num]:
        data_root = pathlib.Path(train_path+'/input')
        traj_length = len(list(data_root.glob('*')))
        total_train_data += traj_length

        for traj_i in range(traj_length):
            # load input and labels
            col_names=['q1', 'q2', 'q3','q4', 'q5', 'q6'] 
            data = read_csv(train_path+"/input/"+str(traj_i)+".csv", names=col_names)
            data_labels = read_csv(train_path+"/label/"+str(traj_i)+".csv", names=col_names)
            this_input = np.array([])
            this_label = np.array([])
            for ji in range(JN):
                curve_q=np.array(data['q'+str(ji+1)].tolist())
                curve_q_labels=np.array(data_labels['q'+str(ji+1)].tolist())
                this_input = np.append(this_input,curve_q)
                # this_input = np.append(this_input,np.deg2rad(curve_q))
                this_label = np.append(this_label,curve_q_labels-curve_q[Tf+1:])
                # print(curve_q_labels-curve_q[Tf+1:])
            # this_label *= 100
            train_inputs.append(this_input)
            train_labels.append(this_label)
            
    
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
            # this_label *= 100
            test_inputs.append(this_input)
            test_labels.append(this_label)

    # print("j1 inputs len:",len(test_inputs_all[0]))
    # print("j1 labels len:",len(test_labels_all[0]))

    train_inputs = tf.convert_to_tensor(np.array(train_inputs), dtype=tf.float64)
    train_labels = tf.convert_to_tensor(np.array(train_labels), dtype=tf.float64)
    test_inputs = tf.convert_to_tensor(np.array(test_inputs), dtype=tf.float64)
    test_labels = tf.convert_to_tensor(np.array(test_labels), dtype=tf.float64)

    print(train_inputs.shape)
    print(train_labels.shape)
    print(test_inputs.shape)
    print(test_labels.shape)

    print("Total Train Segments:",total_train_data)
    print("Total Test Segments:",total_test_data)

    return train_inputs,train_labels,test_inputs,test_labels

# prepare data
data_path = '/media/eric/Transcend/Motion-Primitive-Planning/simulation/traj_prediction/data/data_L_z10_split'
data_root = pathlib.Path(data_path)
all_data_paths = list(data_root.glob('*'))
all_data_paths = sorted([str(path) for path in all_data_paths])
# print(all_data_paths)
train_traj_num = 120
test_traj_num = 30
# train_traj_num = 1
# test_traj_num = 1
train_inputs,train_labels,test_inputs,test_labels = load_data(all_data_paths,train_traj_num,test_traj_num)

train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)).batch(batch_size)

# loss object and optimizer
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')

# train function
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(inputs, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # print(loss)
    train_loss(labels, predictions)

@tf.function
def test_step(inputs, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(inputs, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(labels, predictions)

train_loss_save = []
test_loss_save = []

checkpoint_path = "models/{epoch:04d}.ckpt"
save_weight_every = 4
for epoch in range(EPOCHS):

    print("Epoch:",epoch+1)

    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

    for inputs, labels in train_ds:
        train_step(inputs, labels)

    for test_inputs, test_labels in test_ds:
        test_step(test_inputs, test_labels)

    print(
    f'Loss: {train_loss.result()}, '
    f'Test Loss: {test_loss.result()}, '
    )
    print("---------------------")
    train_loss_save.append(train_loss.result().numpy())
    test_loss_save.append(test_loss.result().numpy())

    if epoch % save_weight_every == 0:
        model.save_weights(checkpoint_path.format(epoch=epoch))
    print("======================================")

    with open('train_loss_save.npy','wb') as f:
        np.save(f,np.array(train_loss_save))
    with open('test_loss_save.npy','wb') as f:
        np.save(f,np.array(test_loss_save))