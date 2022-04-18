# 比较预测值和真实值的全场对比
import numpy as np
from pinn_model import *
import pandas as pd
import matplotlib.pyplot as plt
# 网格区域和时间区域信息
compare_time_sec = np.array([0.0, 19.8])
time_interval = 0.1
whole_field = np.array([[1.0, 8.0], [-2, 2]])
small_hole = np.array([[2.5, 3.5], [-0.5, 0.5]])
large_hole = np.array([[2.0, 4.0], [-1, 1]])
small_truncation = np.array([[2.5, 3.5], [-2, 2]])
large_truncation = np.array([[2.0, 4.0], [-2, 2]])
# 预测数据和真实数据加载路径
filename_load_model = './data_3000epoch_cosine/large_trunc/NS_model_train.pt'
filename_data = './cylinder_nektar_wake.mat'
hole = whole_field
# 预测数据获取（预测）
# feature_mat(2,6)(第一行为最大值)(x,y,t,u,v,p)
x, y, t, u, v, p, feature_mat = read_data_hole(filename_data, hole)
data_stack = np.concatenate((x, y, t, u, v, p), axis=1)
del x, y, t, u, v, p
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
pinn_net = PINN_Net(layer_mat)
pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))


# 比较指定时间序列的真实值和预测值-L2_norm
def predict_at_select_time_series(lower_time, upper_time, data_stack, pinn_example, interval=0.1):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)
    min_data = np.min(data_stack, 0).reshape(1, data_stack.shape[1])
    max_data = np.max(data_stack, 0).reshape(1, data_stack.shape[1])
    # 保留数据集中不重复的坐标
    x = np.unique(x).reshape(-1, 1)
    y = np.unique(y).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x, y)
    n = int((upper_time - lower_time) / interval) + 1
    time_lists = np.linspace(lower_time, upper_time, n)
    L2_uvp = np.empty((0, 3), dtype=float)
    for select_time in time_lists:
        select_time = round(float(select_time), 3)
        index_time = np.where(t == select_time)[0]
        u_selected = u[index_time]
        v_selected = v[index_time]
        p_selected = p[index_time]
        x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
        y_flatten = np.ndarray.flatten(mesh_y).reshape(-1, 1)
        t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

        x_selected = torch.tensor(x_flatten, requires_grad=True, dtype=torch.float32).to(device)
        y_selected = torch.tensor(y_flatten, requires_grad=True, dtype=torch.float32).to(device)
        t_selected = torch.tensor(t_flatten, requires_grad=True, dtype=torch.float32).to(device)
        u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_uv_identification(x_selected,
                                                                                                   y_selected,
                                                                                                   t_selected,
                                                                                                   pinn_example,
                                                                                                   lam1=1.0, lam2=0.01)
        u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
        v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
        p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
        u_selected = u_selected.reshape(mesh_x.shape)
        v_selected = v_selected.reshape(mesh_x.shape)
        p_selected = p_selected.reshape(mesh_x.shape)
        L2_uvp_at_moment = L2_norm_at_moment(u_predict, v_predict, p_predict, u_selected, v_selected, p_selected)
        L2_uvp = np.append(L2_uvp, L2_uvp_at_moment, axis=0)
    return L2_uvp


# L2范数对比(除以节点数)
def L2_norm_at_moment(u_predict, v_predict, p_predict, u_selected, v_selected, p_selected):
    bias_p = np.average(p_selected-p_predict)
    u_predict = u_predict.reshape(-1, 1)
    v_predict = v_predict.reshape(-1, 1)
    p_predict = p_predict.reshape(-1, 1)+bias_p
    u_selected = u_selected.reshape(-1, 1)
    v_selected = v_selected.reshape(-1, 1)
    p_selected = p_selected.reshape(-1, 1)
    L2_u_mid = np.abs((u_predict-u_selected)/u_selected)
    L2_u = np.sqrt(np.sum(L2_u_mid * L2_u_mid, axis=0, keepdims=True))/u_selected.shape[0]
    L2_v_mid = np.abs((v_predict-v_selected)/v_selected)
    L2_v = np.sqrt(np.sum(L2_v_mid * L2_v_mid, axis=0, keepdims=True))/v_selected.shape[0]
    L2_p_mid = np.abs((p_predict-p_selected)/p_selected)
    L2_p = np.sqrt(np.sum(L2_p_mid * L2_p_mid, axis=0, keepdims=True))/p_selected.shape[0]
    L2_uvp_at_moment = np.concatenate((L2_u, L2_v, L2_p), axis=1)
    return L2_uvp_at_moment


# 比较指定时间序列的真实值和预测值-L2_norm(除以uvp的最大值)
def predict_at_select_time_series_modified(lower_time, upper_time, data_stack, pinn_example, feature_mat, interval=0.1):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)
    min_data = np.min(data_stack, 0).reshape(1, data_stack.shape[1])
    max_data = np.max(data_stack, 0).reshape(1, data_stack.shape[1])
    # 保留数据集中不重复的坐标
    x = np.unique(x).reshape(-1, 1)
    y = np.unique(y).reshape(-1, 1)
    mesh_x, mesh_y = np.meshgrid(x, y)
    n = int((upper_time - lower_time) / interval) + 1
    time_lists = np.linspace(lower_time, upper_time, n)
    L2_uvp = np.empty((0, 3), dtype=float)
    for select_time in time_lists:
        select_time = round(float(select_time), 3)
        index_time = np.where(t == select_time)[0]
        u_selected = u[index_time]
        v_selected = v[index_time]
        p_selected = p[index_time]
        x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)
        y_flatten = np.ndarray.flatten(mesh_y).reshape(-1, 1)
        t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time

        x_selected = torch.tensor(x_flatten, requires_grad=True, dtype=torch.float32).to(device)
        y_selected = torch.tensor(y_flatten, requires_grad=True, dtype=torch.float32).to(device)
        t_selected = torch.tensor(t_flatten, requires_grad=True, dtype=torch.float32).to(device)
        u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_uv_identification(x_selected,
                                                                                                   y_selected,
                                                                                                   t_selected,
                                                                                                   pinn_example,
                                                                                                   lam1=1.0, lam2=0.01)
        u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
        v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
        p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
        u_selected = u_selected.reshape(mesh_x.shape)
        v_selected = v_selected.reshape(mesh_x.shape)
        p_selected = p_selected.reshape(mesh_x.shape)
        feature_matrix = feature_mat.data.numpy()
        L2_uvp_at_moment = L2_norm_at_moment_modified(u_predict, v_predict, p_predict, u_selected, v_selected, p_selected,feature_matrix)
        L2_uvp = np.append(L2_uvp, L2_uvp_at_moment, axis=0)
    return L2_uvp


# L2范数对比(除以节点数)(除以各自值的最大值)
def L2_norm_at_moment_modified(u_predict, v_predict, p_predict, u_selected, v_selected, p_selected, feature_mat):
    bias_p = np.average(p_selected-p_predict)
    u_predict = u_predict.reshape(-1, 1)
    v_predict = v_predict.reshape(-1, 1)
    p_predict = p_predict.reshape(-1, 1)+bias_p
    u_selected = u_selected.reshape(-1, 1)
    v_selected = v_selected.reshape(-1, 1)
    p_selected = p_selected.reshape(-1, 1)
    L2_u_mid = np.abs((u_predict-u_selected)/feature_mat[0, 3])
    L2_u = np.sqrt(np.sum(L2_u_mid * L2_u_mid, axis=0, keepdims=True))/u_selected.shape[0]
    L2_v_mid = np.abs((v_predict-v_selected)/feature_mat[0, 4])
    L2_v = np.sqrt(np.sum(L2_v_mid * L2_v_mid, axis=0, keepdims=True))/v_selected.shape[0]
    L2_p_mid = np.abs((p_predict-p_selected)/feature_mat[0, 5])
    L2_p = np.sqrt(np.sum(L2_p_mid * L2_p_mid, axis=0, keepdims=True))/p_selected.shape[0]
    L2_uvp_at_moment = np.concatenate((L2_u, L2_v, L2_p), axis=1)
    return L2_uvp_at_moment


# 求出相对L2范数并作图
L2_uvp = predict_at_select_time_series_modified(compare_time_sec[0], compare_time_sec[1], data_stack, pinn_net,feature_mat, interval=0.1)
x = np.linspace(1,L2_uvp.shape[0],L2_uvp.shape[0]).reshape(-1,1)
u_l2 = L2_uvp[:, 0].reshape(-1,1)
v_l2 = L2_uvp[:, 1].reshape(-1,1)
p_l2 = L2_uvp[:, 2].reshape(-1,1)
save_data = pd.DataFrame(L2_uvp)
save_data.to_excel("large_trunc_uvp.xlsx")
plt.figure()
plt.title('Relative_L2_u')  # 折线图标题
plt.xlabel('time')  # x轴标题
plt.ylabel('L2_u')  # y轴标题
plt.plot(x, u_l2, marker='o', markersize=0)  # 绘制折线图，添加数据点，设置点的大小
plt.show()
plt.figure()
plt.title('Relative_L2_v')  # 折线图标题
plt.xlabel('time')  # x轴标题
plt.ylabel('L2_v')  # y轴标题
plt.plot(x, v_l2, marker='o', markersize=0)  # 绘制折线图，添加数据点，设置点的大小
plt.show()
plt.figure()
plt.title('Relative_L2_p')  # 折线图标题
plt.xlabel('time')  # x轴标题
plt.ylabel('L2_p')  # y轴标题
plt.plot(x, p_l2, marker='o', markersize=0)  # 绘制折线图，添加数据点，设置点的大小
plt.show()