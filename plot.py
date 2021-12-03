# 将预测结果和真实结果进行可视化对比
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pinn_model import *
import imageio
mpl.use("Agg")

percent = 100
filename_load_model = './part_train_file/'+str(percent)+'percent_data/NS_model_train.pt'
filename_data = './cylinder_nektar_wake.mat'
x, y, t, u, v, p, N, T, feature_mat = read_data(filename_data)
data_stack = np.concatenate((x, y, t, u, v, p), axis=1)
del x, y, t, u, v, p
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
pinn_net = PINN_Net(layer_mat)
pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))


# 选定时刻的对比
def compare_at_select_time(select_time, data_stack, pinn_example):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)
    min_data = np.min(data_stack, 0).reshape(1, data_stack.shape[1])
    max_data = np.max(data_stack, 0).reshape(1, data_stack.shape[1])
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
    plot_compare(u_selected, u_predict, select_time, name='u', min_value=min_data[0, 3], max_value=max_data[0, 3])
    plot_compare(v_selected, v_predict, select_time, name='v', min_value=min_data[0, 4], max_value=max_data[0, 4])
    plot_compare(p_selected, p_predict, select_time, name='p', min_value=min_data[0, 5], max_value=max_data[0, 5])
    print('ok')


# 选定时间序列的对比
def compare_at_select_time_simple_norm(select_time, data_stack,feature_mat, pinn_example):
    x = data_stack[:, 0].copy().reshape(-1, 1)
    y = data_stack[:, 1].copy().reshape(-1, 1)
    t = data_stack[:, 2].copy().reshape(-1, 1)
    u = data_stack[:, 3].copy().reshape(-1, 1)
    v = data_stack[:, 4].copy().reshape(-1, 1)
    p = data_stack[:, 5].copy().reshape(-1, 1)
    min_data = np.min(data_stack, 0).reshape(1, data_stack.shape[1])
    max_data = np.max(data_stack, 0).reshape(1, data_stack.shape[1])
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
    x_flatten = np.ndarray.flatten(mesh_x).reshape(-1, 1)/feature_mat.data.numpy()[0, 0]
    y_flatten = np.ndarray.flatten(mesh_y).reshape(-1, 1)/feature_mat.data.numpy()[0, 1]
    t_flatten = np.ones((x_flatten.shape[0], 1)) * select_time/feature_mat.data.numpy()[0, 2]

    x_selected = torch.tensor(x_flatten, requires_grad=True, dtype=torch.float32).to(device)
    y_selected = torch.tensor(y_flatten, requires_grad=True, dtype=torch.float32).to(device)
    t_selected = torch.tensor(t_flatten, requires_grad=True, dtype=torch.float32).to(device)
    del x_flatten, y_flatten, t_flatten
    u_predict, v_predict, p_predict,f_equation_c, f_equation_x, f_equation_y = f_equation_inverse_simple_norm(x_selected, y_selected, t_selected,feature_mat,
                                                                                     pinn_example)
    u_predict = u_predict * feature_mat[0, 3]
    v_predict = v_predict * feature_mat[0, 4]
    p_predict = p_predict * feature_mat[0, 5]
    # 画图
    u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
    v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
    p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
    u_selected = u_selected.reshape(mesh_x.shape)
    v_selected = v_selected.reshape(mesh_x.shape)
    p_selected = p_selected.reshape(mesh_x.shape)
    plot_compare(u_selected, u_predict, select_time, name='u', min_value=min_data[0, 3], max_value=max_data[0, 3])
    plot_compare(v_selected, v_predict, select_time, name='v', min_value=min_data[0, 4], max_value=max_data[0, 4])
    plot_compare(p_selected, p_predict, select_time, name='p', min_value=min_data[0, 5], max_value=max_data[0, 5])
    print('ok')


def plot_compare(q_selected, q_predict, select_time, min_value, max_value, name='q'):
    fig_q = plt.figure(figsize=(10, 4))
    v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    plt.subplot(1, 2, 1)
    plt.imshow(q_selected, cmap='jet', norm=v_norm)
    plt.title("True_value:" + name + "(x,y,t) at t=" + str(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(q_predict, cmap='jet', norm=v_norm)
    plt.title("Predict_value:" + name + "(x,y,t) at t=" + str(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.show()


# 选定时间序列的对比
def compare_at_select_time_series(lower_time, upper_time, data_stack, pinn_example, interval=0.1):
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
        u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_inverse(x_selected, y_selected,
                                                                                         t_selected,
                                                                                         pinn_example)
        u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
        v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
        p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
        u_selected = u_selected.reshape(mesh_x.shape)
        v_selected = v_selected.reshape(mesh_x.shape)
        p_selected = p_selected.reshape(mesh_x.shape)
        plot_compare_time_series(u_selected, u_predict, select_time, name='u', min_value=min_data[0, 3],
                                 max_value=max_data[0, 3])
        plot_compare_time_series(v_selected, v_predict, select_time, name='v', min_value=min_data[0, 4],
                                 max_value=max_data[0, 4])
        plot_compare_time_series(p_selected, p_predict, select_time, name='p', min_value=min_data[0, 5],
                                 max_value=max_data[0, 5])
        del u_selected, v_selected, p_selected, x_flatten, y_flatten, t_flatten
        del x_selected, y_selected, t_selected, u_predict, v_predict, p_predict
        plt.close('all')


def plot_compare_time_series(q_selected, q_predict, select_time, min_value, max_value, name='q'):
    plt.cla()
    v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(q_selected, cmap='jet', norm=v_norm)
    plt.title("True_" + name + "_value:" + " t=" + "{:.2f}".format(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(q_predict, cmap='jet', norm=v_norm)
    plt.title("Predict_" + name + "_value:" + " t=" + "{:.2f}".format(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.savefig('./gif_make/' + 'time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')


# 预测值和真实值的相减
def subtract_at_select_time_series(lower_time, upper_time, data_stack, pinn_example, interval=0.1):
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
        u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_inverse(x_selected, y_selected,
                                                                                         t_selected,
                                                                                         pinn_example)
        u_predict = u_predict.data.numpy().reshape(mesh_x.shape)
        v_predict = v_predict.data.numpy().reshape(mesh_x.shape)
        p_predict = p_predict.data.numpy().reshape(mesh_x.shape)
        u_selected = u_selected.reshape(mesh_x.shape)
        v_selected = v_selected.reshape(mesh_x.shape)
        p_selected = p_selected.reshape(mesh_x.shape)
        u_subtract = (u_predict-u_selected)/u_selected
        v_subtract = (v_predict - v_selected)/v_selected
        p_subtract = (p_predict - p_selected)/p_selected
        plot_subtract_time_series(u_subtract, select_time, name='u')
        plot_subtract_time_series(v_subtract, select_time, name='v')
        plot_subtract_time_series(p_subtract, select_time, name='p')
        del u_selected, v_selected, p_selected, x_flatten, y_flatten, t_flatten
        del x_selected, y_selected, t_selected, u_predict, v_predict, p_predict
        plt.close('all')


def plot_subtract_time_series(q_subtract, select_time, name='q'):
    plt.cla()
    # min_value = np.min(q_subtract)
    # max_value = np.max(q_subtract)
    min_value = 0.0
    max_value = 1.0
    v_norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    plt.figure(figsize=(5, 4))
    plt.imshow(q_subtract, cmap='jet', norm=v_norm)
    plt.title("100Subtract_" + name + "_value:" + " t=" + "{:.2f}".format(select_time))
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.colorbar()
    plt.savefig('./gif_make/' + 'time' + "{:.2f}".format(select_time) + name + '.png')
    plt.close('all')



# 制作gif图
def make_flow_gif(lower_time, upper_time, interval=0.1, name='q', fps_num=5):
    gif_images = []
    n = int((upper_time - lower_time) / interval) + 1
    time_lists = np.linspace(lower_time, upper_time, n)
    for select_time in time_lists:
        select_time = round(float(select_time), 3)
        gif_images.append(imageio.imread('./gif_make/' + 'time' + "{:.2f}".format(select_time) + name + '.png'))
    imageio.mimsave(('100'+name+'.gif'), gif_images, fps=fps_num)


# choose_time = 4.8
# # compare_at_select_time(choose_time, data_stack, pinn_net)
# compare_at_select_time_simple_norm(choose_time, data_stack,feature_mat, pinn_net)


start_time = 0.0
end_time = 19.0
interval = 0.1
# compare_at_select_time_series(start_time, end_time, data_stack, pinn_net, interval=0.1)
subtract_at_select_time_series(start_time, end_time, data_stack, pinn_net, interval=0.1)
print("image done")
make_flow_gif(start_time, end_time, interval=0.1, name='u', fps_num=20)
make_flow_gif(start_time, end_time, interval=0.1, name='v', fps_num=20)
make_flow_gif(start_time, end_time, interval=0.1, name='p', fps_num=20)
print("gif done")

