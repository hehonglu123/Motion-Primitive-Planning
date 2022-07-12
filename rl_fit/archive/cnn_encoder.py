import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import os

from curve_normalization import PCA_normalization


def read_data(dir_path, data_size=None):
    all_data = []
    col_names = ['X', 'Y', 'Z', 'direction_x', 'direction_y', 'direction_z']

    for file in os.listdir(dir_path):
        if data_size is not None and len(all_data) >= data_size:
            break

        file_path = dir_path + os.sep + file
        data = pd.read_csv(file_path, names=col_names)
        curve_x = data['X'].tolist()
        curve_y = data['Y'].tolist()
        curve_z = data['Z'].tolist()
        curve = np.vstack((curve_x, curve_y, curve_z)).T
        all_data.append(curve)

    return all_data


def curve_to_train_data(curve_data, points, ids):
    x_train = []
    for i in range(len(ids)):
        point = points[i]
        curve = curve_data[ids[i]]
        normalized_curve = PCA_normalization(curve[point:])
        x_train.append(normalized_curve.T)
    x_train = np.array(x_train)

    return x_train


def dataset_loader(x_data, y_data, batch_size=16, shuffle=False):
    data_size = x_data.shape[0]
    n_batch = (np.ceil(data_size / batch_size)).astype(int)

    data_index = np.arange(data_size)
    if shuffle:
        np.random.shuffle(data_index)
    loader = []
    for i in range(n_batch):
        batch_index = data_index[i*batch_size:min(data_size, (i+1)*batch_size)]
        x_data_batch = x_data[batch_index, :, :]
        y_data_batch = y_data[batch_index]
        loader.append((x_data_batch, y_data_batch))
    return loader


class CNN(nn.Module):
    def __init__(self, output_dim=2):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=200, stride=50)
        self.linear1 = nn.Linear(in_features=136, out_features=128)
        # self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)

        x = self.relu(self.linear1(x))
        # x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=2):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=200, stride=50)
        self.linear1 = nn.Linear(in_features=136, out_features=128)
        # self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = x / torch.norm(x)
        return x


def train_cnn(model, x_data, y_data, optimizer, criteria, n_epoch=100, batch_size=16):
    model.train()

    for epoch in range(n_epoch):
        epoch_accuracy = 0
        loader = dataset_loader(x_data, y_data, shuffle=True)
        for x_batch, y_batch in loader:
            x_batch_tensor = torch.from_numpy(x_batch).float()
            y_batch_tensor = torch.LongTensor(y_batch)
            output = F.softmax(model(x_batch_tensor), dim=1)

            loss = criteria(output, y_batch_tensor)
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            accuracy = pred.eq(y_batch_tensor.view_as(pred)).sum().item() / y_batch_tensor.shape[0]
            epoch_accuracy += accuracy
        epoch_accuracy /= len(loader)
        if (epoch + 1) % 10 == 0:
            print("Epoch {} / {}: Accuracy: {:.2f}".format(epoch + 1, n_epoch, epoch_accuracy))


def load_encoder(model_pth):
    cnn_encoder = CNNEncoder()
    cnn_encoder.load_state_dict(torch.load(model_pth))
    return cnn_encoder


def main():
    cnn_model_path = '../cnn_model/cnn_model.pth'
    cnn_encoder = load_encoder(cnn_model_path)
    dataset = read_data('../train_data/base', data_size=1)

    for data in dataset:
        cnn_encoder.eval()
        print(data.shape)
        data_normalized = PCA_normalization(data)
        print(data_normalized.shape)
        data_tensor = torch.from_numpy(np.array([data_normalized.T])).float()
        output = cnn_encoder(data_tensor)
        print(output.shape)
        feature = output.reshape(-1)
        print(feature.shape)
        flattened = output.flatten()
        print(flattened.shape)


if __name__ == '__main__':
    main()
