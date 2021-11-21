# 将预测结果和真实结果进行可视化对比
import matplotlib.pyplot as plt
import numpy as np
from pinn_model import *
filename_save_model = './NS_model_train.pt'

x, y, t, u, v, p, N, T = read_data(filename_data)
data_stack = np.concatenate((x, y, t, u, v, p), axis=1)
del x, y, t, u, v, p
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
pinn_net = PINN_Net(layer_mat)
pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))


def compare_at_select_time(select_time, data_stack, pinn_example):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)
    index_time = np.where(t == select_time)[0]
    # 保留数据集中不重复的坐标
    x = np.unique(x).reshape(-1, 1)
    y = np.unique(y).reshape(-1, 1)
    # 选出指定时刻的u,v,p
    u_selected = u[index_time]
    v_selected = v[index_time]
    p_selected = p[index_time]
    # 给出x,y的网格点
    mesh_x, mesh_y = np.meshgrid(x, y)
    x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
    y_flatten = np.ndarray.flatten(mesh_y).reshape(-1, 1)
    t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

    x_selected = torch.tensor(x_flatten, requires_grad=True, dtype=torch.float32).to(device)
    y_selected = torch.tensor(y_flatten, requires_grad=True, dtype=torch.float32).to(device)
    t_selected = torch.tensor(t_flatten, requires_grad=True, dtype=torch.float32).to(device)
    del x_flatten, y_flatten, t_flatten
    u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_inverse(x_selected, y_selected, t_selected,
                                                                                     pinn_example)
    # 画图
    u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
    v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
    p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
    u_selected = u_selected.reshape(mesh_x.shape)
    v_selected = v_selected.reshape(mesh_x.shape)
    p_selected = p_selected.reshape(mesh_x.shape)
    plot_compare(u_selected, u_predict, select_time)
    plot_compare(v_selected, v_predict, select_time)
    plot_compare(p_selected, p_predict, select_time)
    print('ok')


def plot_compare(q_selected, q_predict, select_time):
    fig_q = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(q_selected, cmap='jet')
    plt.title("True_value:u(x,y,t) at t=" + str(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(q_predict, cmap='jet')
    plt.title("Predict_value:u(x,y,t) at t=" + str(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.show()


choose_time = 4.8
compare_at_select_time(choose_time, data_stack, pinn_net)
