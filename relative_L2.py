import numpy as np
import scipy
import torch
import os
from pinn_model import *
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def load_full_points(filename):
    # for each process, load data points within the extended boundary
    # 每一个进程读取相应区域延拓边界内的数据点
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
    x_ts = torch.tensor(x, dtype=torch.float32)
    y_ts = torch.tensor(y, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    u_ts = torch.tensor(u, dtype=torch.float32)
    v_ts = torch.tensor(v, dtype=torch.float32)
    p_ts = torch.tensor(p, dtype=torch.float32)
    return x_ts, y_ts, t_ts, u_ts, v_ts, p_ts, low_bound, up_bound


def compute_L2_norm(filename_train_data, filename_raw_data, filename_predict, norm_status="no_norm"):
    # 预测值
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, low_bound, up_bound = load_full_points(filename_raw_data)
    x_pre = x_raw.clone().detach().requires_grad_(True).to(device)
    y_pre = y_raw.clone().detach().requires_grad_(True).to(device)
    t_pre = t_raw.clone().detach().requires_grad_(True).to(device)
    data_mean, data_std = load_data_feature(filename_train_data)
    pinn_net = PINN_Net(layer_mat, data_mean, data_std, device)
    pinn_net = pinn_net.to(device)
    filename_load_model = filename_predict + '/NS_model_train.pt'
    pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
    if norm_status == "no_norm":
        u_pre, v_pre, p_pre = pinn_net.predict(x_pre, y_pre, t_pre)
    else:
        u_pre, v_pre, p_pre = pinn_net.predict_inner_norm(x_pre, y_pre, t_pre)
    u_raw_mat = u_raw.numpy()
    v_raw_mat = v_raw.numpy()
    p_raw_mat = p_raw.numpy()
    u_pre_mat = u_pre.cpu().detach().numpy()
    v_pre_mat = v_pre.cpu().detach().numpy()
    p_pre_mat = p_pre.cpu().detach().numpy()
    # 处理数据
    L2_u = (np.linalg.norm(u_pre_mat - u_raw_mat) / np.linalg.norm(u_raw_mat)).reshape(-1, 1)
    L2_v = (np.linalg.norm(v_pre_mat - v_raw_mat) / np.linalg.norm(v_raw_mat)).reshape(-1, 1)
    L2_p = (np.linalg.norm(p_pre_mat - p_raw_mat) / np.linalg.norm(p_raw_mat)).reshape(-1, 1)
    L2_uvp_at_moment = np.concatenate((L2_u, L2_v, L2_p), axis=1)
    return L2_uvp_at_moment


def L2_norm_at_moment(u_predicted, v_predicted, p_predicted, u_selected, v_selected, p_selected):
    L2_u = (np.linalg.norm(u_predicted - u_selected) / np.linalg.norm(u_selected)).reshape(-1, 1)
    L2_v = (np.linalg.norm(v_predicted - v_selected) / np.linalg.norm(v_selected)).reshape(-1, 1)
    L2_p = (np.linalg.norm(p_predicted - p_selected) / np.linalg.norm(p_selected)).reshape(-1, 1)
    L2_uvp_at_moment = np.concatenate((L2_u, L2_v, L2_p), axis=1)
    return L2_uvp_at_moment


def rearrange_folders(matching_folders_list):
    def compare_by_number(string):
        # 提取字符串中的数字部分，并将其转换为整数进行比较
        number = int(string[4:])
        return number

    rearranged_list = sorted(matching_folders_list, key=compare_by_number)
    return rearranged_list


if __name__ == "__main__":
    layer_mat_1 = [3, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 3]  # 网络结构 neural network
    layer_mat_2 = [3, 80, 80, 80, 80, 80, 3]  # 网络结构 neural network
    layer_mat_3 = [3, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 3]  # 网络结构 neural network
    layer_mat = layer_mat_1
    filename_raw_data = './cylinder_Re3900_ke_all_100snaps.mat'
    filename_train_data = './cylinder_Re3900_36points_100snaps.mat'
    name_of_csv_set = ['exp_3900ke_1', 'exp_3900ke_2', 'exp_3900ke_3', 'exp_3900ke_4', 'exp_3900ke_5']
    for name in name_of_csv_set:
        folder_path = './data/' + name + '/write/'# 文件夹路径
        prefix = 'step'  # 文件夹名前缀
        # 获取文件夹下所有文件夹名
        folder_list = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]
        # 筛选以指定前缀开头的文件夹名
        matching_folders = [folder for folder in folder_list if folder.startswith(prefix)]
        matching_folders = rearrange_folders(matching_folders)
        print(matching_folders)
        norm_status = "no_norm"
        # norm_status = "inner_norm"
        L2_set = np.empty((0, 3))
        for dir_name in matching_folders:
            file_predict = folder_path + dir_name
            L2_norm = compute_L2_norm(filename_train_data, filename_raw_data, file_predict, norm_status)
            L2_set = np.append(L2_set, L2_norm, axis=0)
        # 存储L2范数信息
        df = pd.DataFrame(L2_set, columns=["U", "V", "P"])
        if not os.path.exists('./data_csv'):
            os.makedirs('./data_csv')
        filename_csv = './data_csv/non_dimensional_' + name + '.csv'
        df.to_csv(filename_csv, index=False)
        print('ok')