"""
core functions
核心函数
"""
import numpy as np
import scipy
import pandas as pd
from pinn_model import *

# load data feature for inner normalization method
# 加载数据均值和标准差，适用于inner norm 方法
def load_data_feature(filename):
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    temp = np.concatenate((x, y, t), 1)
    data_mean = np.mean(temp, axis=0).reshape(1, -1)
    data_std = np.std(temp, axis=0).reshape(1, -1)
    return data_mean, data_std


# load data points
# 网络加载数据点
def load_data_points(filename):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    u = stack[:, 3].reshape(-1, 1)
    v = stack[:, 4].reshape(-1, 1)
    p = stack[:, 5].reshape(-1, 1)
    low_bound = np.array([np.min(x), np.min(y), np.min(t)]).reshape(1, -1)
    up_bound = np.array([np.max(x), np.max(y), np.max(t)]).reshape(1, -1)
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.min(temp, 0)
    feature_mat[1, :] = np.max(temp, 0)
    x_ts = torch.tensor(x, dtype=torch.float32)
    y_ts = torch.tensor(y, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    u_ts = torch.tensor(u, dtype=torch.float32)
    v_ts = torch.tensor(v, dtype=torch.float32)
    p_ts = torch.tensor(p, dtype=torch.float32)
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound


# load collocation points
# 加载方程点——拉丁超立方
def load_equation_points_lhs(low_bound, up_bound, dimension, points):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension, points)
    Eqa_points = torch.from_numpy(eqa_xyzt).float()
    Eqa_points = Eqa_points[torch.randperm(Eqa_points.size(0))]
    return Eqa_points


# split batch and automatically fill batch size
# batch 划分与自动填充
def batch_split(Set, iter_num, dim=0):
    batches = torch.chunk(Set, iter_num, dim=dim)
    # 自动填充
    num_of_batches = len(batches)
    if num_of_batches == 1:
        batches = batches * iter_num
        return batches
    if num_of_batches < iter_num:
        for i in range(iter_num - num_of_batches):
            index = i % num_of_batches
            add_tuple = batches[-(index + 2):-(index + 1)]
            batches = batches + add_tuple
        return batches
    else:
        return batches


# preprocessing before training
# 训练前预处理
# 数据划分batch
def pre_train_loading(filename_data, dimension, N_eqa, batch_size):
    # load data points(only once)
    # 加载真实数据点(仅一次)
    x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts, low_bound, up_bound = load_data_points(filename_data)
    if x_sub_ts.shape[0] > 0:
        data_sub = torch.cat([x_sub_ts, y_sub_ts, t_sub_ts, u_sub_ts, v_sub_ts, p_sub_ts], 1)
        true_dataset = data_sub[torch.randperm(data_sub.size(0))]  # 乱序
    else:
        true_dataset = None
    # load collocation points(only once)
    # 加载方程点(仅一次)
    eqa_points = load_equation_points_lhs(low_bound, up_bound, dimension, N_eqa)
    eqa_points_batches = torch.split(eqa_points, batch_size, dim=0)
    iter_num = len(eqa_points_batches)
    true_dataset_batches = batch_split(true_dataset, iter_num)
    return true_dataset, eqa_points_batches, iter_num, low_bound, up_bound


# train data points, collocation points--with batch training
# 同时训练数据点，方程点———有batch
def train_data_whole(inner_epochs, pinn_example, optimizer_all, scheduler_all, iter_num, true_dataset,
                     Eqa_points_batches, Re, weight_data, weight_eqa, EPOCH, debug_key, device):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    x_train = true_dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = true_dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = true_dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = true_dataset[:, 3].reshape(-1, 1).to(device)
    v_train = true_dataset[:, 4].reshape(-1, 1).to(device)
    p_train = true_dataset[:, 5].reshape(-1, 1).to(device)
    for epoch in range(inner_epochs):
        for batch_iter in range(iter_num):
            optimizer_all.zero_grad()
            x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
            y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
            t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
            mse_data = pinn_example.data_mse(x_train, y_train, t_train, u_train, v_train, p_train)
            mse_equation = pinn_example.equation_mse_dimensionless(x_eqa, y_eqa, t_eqa, Re)
            # calculate loss
            # 计算损失函数
            loss = weight_data * mse_data + weight_eqa * mse_equation
            loss.backward()
            optimizer_all.step()
            with torch.autograd.no_grad():
                loss_sum = loss.cpu().data.numpy()
                loss_data = mse_data.cpu().data.numpy()
                loss_eqa = mse_equation.cpu().data.numpy()
                # print status
                # 输出状态
                if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                    print("EPOCH:", (EPOCH + 1), "  inner_iter:", batch_iter + 1, " Training-data Loss:",
                          round(float(loss.data), 8))
        scheduler_all.step()
    return loss_sum, loss_data, loss_eqa


# train with inner normalization
# 网络内嵌归一化层训练
def train_data_whole_inner_norm(inner_epochs, pinn_example, optimizer_all, scheduler_all, iter_num, true_dataset,
                                Eqa_points_batches, Re, weight_data, weight_eqa, EPOCH, debug_key, device):
    loss_sum = np.array([0.0]).reshape(1, 1)
    loss_data = np.array([0.0]).reshape(1, 1)
    loss_eqa = np.array([0.0]).reshape(1, 1)
    x_train = true_dataset[:, 0].reshape(-1, 1).requires_grad_(True).to(device)
    y_train = true_dataset[:, 1].reshape(-1, 1).requires_grad_(True).to(device)
    t_train = true_dataset[:, 2].reshape(-1, 1).requires_grad_(True).to(device)
    u_train = true_dataset[:, 3].reshape(-1, 1).to(device)
    v_train = true_dataset[:, 4].reshape(-1, 1).to(device)
    p_train = true_dataset[:, 5].reshape(-1, 1).to(device)
    for epoch in range(inner_epochs):
        for batch_iter in range(iter_num):
            optimizer_all.zero_grad()
            x_eqa = Eqa_points_batches[batch_iter][:, 0].reshape(-1, 1).requires_grad_(True).to(device)
            y_eqa = Eqa_points_batches[batch_iter][:, 1].reshape(-1, 1).requires_grad_(True).to(device)
            t_eqa = Eqa_points_batches[batch_iter][:, 2].reshape(-1, 1).requires_grad_(True).to(device)
            mse_data = pinn_example.data_mse_inner_norm(x_train, y_train, t_train, u_train, v_train, p_train)
            mse_equation = pinn_example.equation_mse_dimensionless_inner_norm(x_eqa, y_eqa, t_eqa, Re)
            # calculate loss
            # 计算损失函数
            loss = weight_data * mse_data + weight_eqa * mse_equation
            loss.backward()
            optimizer_all.step()
            with torch.autograd.no_grad():
                loss_sum = loss.cpu().data.numpy()
                loss_data = mse_data.cpu().data.numpy()
                loss_eqa = mse_equation.cpu().data.numpy()
                # print status
                # 输出状态
                if (batch_iter + 1) % iter_num == 0 and debug_key == 1:
                    print("EPOCH:", (EPOCH + 1), "  inner_iter:", batch_iter + 1, " Training-data Loss:",
                          round(float(loss.data), 8))
        scheduler_all.step()
    return loss_sum, loss_data, loss_eqa


# record loss
# 记录loss
def record_loss_local(loss_sum, loss_data, loss_eqa, filename_loss):
    loss_sum_value = loss_sum.reshape(1, 1)
    loss_data_value = loss_data.reshape(1, 1)
    loss_eqa_value = loss_eqa.reshape(1, 1)
    loss_set = np.concatenate((loss_sum_value, loss_data_value, loss_eqa_value), 1).reshape(1, -1)
    loss_save = pd.DataFrame(loss_set)
    loss_save.to_csv(filename_loss, index=False, header=False, mode='a')
    return loss_set
