import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import numpy as np
import scipy
import torch
import os
from pinn_model import *
import matplotlib.tri as tri

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_time(filename):
    # load t
    # 读取时间数据
    data_mat = scipy.io.loadmat(filename)
    stack = data_mat['stack']  # N*4 (x,y,u,v)
    x = stack[:, 0].reshape(-1, 1)
    y = stack[:, 1].reshape(-1, 1)
    t = stack[:, 2].reshape(-1, 1)
    u = stack[:, 3].reshape(-1, 1)
    v = stack[:, 4].reshape(-1, 1)
    p = stack[:, 5].reshape(-1, 1)
    return  t


def load_data_points(filename):
    # load all data
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


def compare_unsteady(filename_raw_data, filename_predict, start_index, end_index, norm_status="no_norm", mode="compare"):
    # 预测值
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, low_bound, up_bound = load_data_points(filename_raw_data)
    x_pre = x_raw.clone().detach().requires_grad_(True).to(device)
    y_pre = y_raw.clone().detach().requires_grad_(True).to(device)
    t_pre = t_raw.clone().detach().requires_grad_(True).to(device)
    pinn_net = PINN_Net(layer_mat, low_bound, up_bound, device)
    pinn_net = pinn_net.to(device)
    filename_load_model = filename_predict + '/NS_model_train.pt'
    pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
    if norm_status == "no_norm":
        u_pre, v_pre, p_pre = pinn_net.predict(x_pre, y_pre, t_pre)

    else:
        u_pre, v_pre, p_pre = pinn_net.predict_inner_norm(x_pre, y_pre, t_pre)

    # 处理数据
    x_raw_mat = x_raw.numpy()
    y_raw_mat = y_raw.numpy()
    t_raw_mat = t_raw.numpy()
    u_raw_mat = u_raw.numpy()
    v_raw_mat = v_raw.numpy()
    p_raw_mat = p_raw.numpy()
    u_pre_mat = u_pre.cpu().detach().numpy()
    v_pre_mat = v_pre.cpu().detach().numpy()
    p_pre_mat = p_pre.cpu().detach().numpy()
    t_unique = np.unique(t_raw_mat).reshape(-1, 1)
    x_unique = np.unique(x_raw_mat).reshape(-1, 1)
    y_unique = np.unique(y_raw_mat).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x_unique, y_unique)
    time_series = t_unique[start_index:end_index, 0].reshape(-1, 1)
    temp = np.concatenate((x_raw_mat, y_raw_mat, t_raw_mat, u_raw_mat, v_raw_mat, p_raw_mat), 1)
    min_data = np.min(temp, axis=0).reshape(1, -1)
    max_data = np.max(temp, axis=0).reshape(1, -1)
    v_norm_u = mpl.colors.Normalize(vmin=min_data[0, 3], vmax=max_data[0, 3])
    v_norm_v = mpl.colors.Normalize(vmin=min_data[0, 4], vmax=max_data[0, 4])
    v_norm_p = mpl.colors.Normalize(vmin=min_data[0, 5], vmax=max_data[0, 5])
    total_step = end_index - start_index
    count = 0
    for select_time in time_series:
        time = select_time.item()
        index_selected = np.where(t_raw_mat == select_time)[0].reshape(-1, 1)
        u_selected = u_raw_mat[index_selected].reshape(mesh_x.shape)
        v_selected = v_raw_mat[index_selected].reshape(mesh_x.shape)
        p_selected = p_raw_mat[index_selected].reshape(mesh_x.shape)
        u_predicted = u_pre_mat[index_selected].reshape(mesh_x.shape)
        v_predicted = v_pre_mat[index_selected].reshape(mesh_x.shape)
        p_predicted = p_pre_mat[index_selected].reshape(mesh_x.shape)
        if mode == 'compare':
            plot_compare_time_series(mesh_x, mesh_y, u_selected, u_predicted, time, v_norm_u, mode, name='u')
            plot_compare_time_series(mesh_x, mesh_y, v_selected, v_predicted, time, v_norm_v, mode, name='v')
            plot_compare_time_series(mesh_x, mesh_y, p_selected, p_predicted, time, v_norm_p, mode, name='p')
        elif mode == 'subtract':
            plot_subtract_time_series(mesh_x, mesh_y, u_selected, u_predicted, time, mode, name='u',
                                      min_value=min_data[0, 3], max_value=max_data[0, 3])
            plot_subtract_time_series(mesh_x, mesh_y, v_selected, v_predicted, time, mode, name='v',
                                      min_value=min_data[0, 4], max_value=max_data[0, 4])
            plot_subtract_time_series(mesh_x, mesh_y, p_selected, p_predicted, time, mode, name='p',
                                      min_value=min_data[0, 5], max_value=max_data[0, 5] * 2)

        count += 1
        print("Plotting ", count, "/", total_step, " pics")
    return


def plot_compare_time_series(x_mesh, y_mesh, q_selected, q_predict, select_time, v_norm, mode, name='q'):
    plt.cla()
    # v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = ax1.contourf(x_mesh, y_mesh, q_selected, levels=200, cmap='jet', norm=v_norm)
    fig.colorbar(mappable, cax=cbar_ax)
    ax1.set_title("True_" + name + "at " + " t=" + "{:.2f}".format(select_time))
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')
    ax2.contourf(x_mesh, y_mesh, q_predict, levels=200, cmap='jet', norm=v_norm)
    ax2.set_title("Predict_" + name + " at " + " t=" + "{:.2f}".format(select_time))
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    if not os.path.exists('gif_make'):
        os.makedirs('gif_make')
    plt.savefig('./gif_make/' + mode + '--time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')


def plot_subtract_time_series(x_mesh, y_mesh, q_selected, q_predict, select_time, mode, min_value, max_value, name='q'):
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    q_bias = np.abs(q_predict - q_selected)
    plt.figure(figsize=(8, 6))
    plt.contourf(x_mesh, y_mesh, q_bias, levels=np.linspace(0.0, 0.10*max(np.abs(min_value), np.abs(max_value)), 200), extend='both', cmap='jet')
    plt.title("Point wise error " + name, fontdict=font)
    plt.ylabel('Y', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('X', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.colorbar()
    if not os.path.exists('gif_make'):
        os.makedirs('gif_make')
    plt.savefig('./gif_make/' + mode + '--time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')


def make_flow_gif(start_index, end_index, t_predict, mode, name='q', fps_num=5):
    gif_images = []
    t_unique = np.unique(t_predict).reshape(-1, 1)
    time_series = t_unique[start_index:end_index, 0].reshape(-1, 1)
    for select_time in time_series:
        time = select_time.item()
        gif_images.append(imageio.imread('./gif_make/' + mode + '--time' + "{:.2f}".format(time) + name + '.png'))
    imageio.mimsave((mode + name + '.gif'), gif_images, fps=fps_num)


if __name__ == "__main__":
    layer_mat_1 = [3, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 3]  # 网络结构 neural network
    layer_mat_2 = [3, 80, 80, 80, 80, 80, 3]  # 网络结构 neural network
    layer_mat_3 = [3, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 3]  # 网络结构 neural network
    layer_mat = layer_mat_1
    filename_raw_data = './cylinder_Re3900_ke_all_100snaps.mat'
    file_predict = './data/exp_3900ke_1/write/step5000'
    start_index = 0
    end_index = 100
    norm_status = "no_norm"
    mode = 'subtract'
    t_predict = load_time(filename_raw_data)

    compare_unsteady(filename_raw_data, file_predict, start_index, end_index, norm_status, mode)
    make_flow_gif(start_index, end_index, t_predict, mode, name='u', fps_num=10)
    make_flow_gif(start_index, end_index, t_predict, mode, name='v', fps_num=10)
    make_flow_gif(start_index, end_index, t_predict, mode, name='p', fps_num=10)
