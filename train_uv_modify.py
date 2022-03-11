import numpy as np
import torch.optim.lr_scheduler

from pinn_model import *
import pandas as pd
import os

# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

# 稀疏训练-无数据归一化处理
# 训练代码主体
# 稀疏占比
data_portion = 0.005
portion = np.sqrt(data_portion)
t_sec = np.array([0.0, 19.8])
whole_field = np.array([[1.0, 8.0], [-2, 2]])
# 数据点和方程点加载
N_eqa = 1000000
dimension = 2+1
x, y, t, u, v, p, feature_mat = read_data_portion(filename_data, portion)
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
X_random = shuffle_data(x, y, t, u, v, p)
lb = np.array([feature_mat.data.numpy()[1, 0], feature_mat.data.numpy()[1, 1], feature_mat.data.numpy()[1, 2]])
ub = np.array([feature_mat.data.numpy()[0, 0], feature_mat.data.numpy()[0, 1], feature_mat.data.numpy()[0, 2]])
Eqa_points = generate_eqp_rect(lb, ub, dimension, N_eqa)
del x, y, t, u, v, p
# 创建PINN模型实例，并将实例分配至对应设备
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)
# 用以记录各部分损失的列表
losses = np.empty((0, 3), dtype=float)

if os.path.exists(filename_save_model):
    pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
if os.path.exists(filename_loss):
    loss_read = pd.read_csv('loss.csv', header=None)
    losses = loss_read.values
# 优化器和学习率衰减设置
optimizer = torch.optim.Adam(pinn_net.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
epochs = 20000

# 选取batch size 此处也可使用data_loader
ratio = 0.20
batch_size_data = int(ratio*X_random.shape[0])
batch_size_eqa = int(ratio*Eqa_points.shape[0])
inner_iter = int(X_random.size(0) / batch_size_data)
eqa_iter = int(Eqa_points.size(0) / batch_size_eqa)

for epoch in range(epochs):
    for batch_iter in range(inner_iter + 1):
        optimizer.zero_grad()
        # 在全集中随机取batch
        if batch_iter < inner_iter:
            x_train = X_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 0].reshape(batch_size_data, 1).clone().requires_grad_(True).to(device)
            y_train = X_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 1].reshape(batch_size_data, 1).clone().requires_grad_(True).to(device)
            t_train = X_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 2].reshape(batch_size_data, 1).clone().requires_grad_(True).to(device)
            u_train = X_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 3].reshape(batch_size_data, 1).clone().requires_grad_(True).to(device)
            v_train = X_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 4].reshape(batch_size_data, 1).clone().requires_grad_(True).to(device)
            p_train = X_random[batch_iter * batch_size_data:((batch_iter + 1) * batch_size_data), 5].reshape(batch_size_data, 1).clone().requires_grad_(True).to(device)
            x_eqa = Eqa_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 0].reshape(batch_size_eqa, 1).clone().requires_grad_(True).to(device)
            y_eqa = Eqa_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 1].reshape(batch_size_eqa, 1).clone().requires_grad_(True).to(device)
            t_eqa = Eqa_points[batch_iter * batch_size_eqa:((batch_iter + 1) * batch_size_eqa), 2].reshape(batch_size_eqa, 1).clone().requires_grad_(True).to(device)
        elif batch_iter == inner_iter:
            if X_random[batch_iter * batch_size_data:, 0].reshape(-1, 1).shape[0] == 0:
                continue
            else:
                x_train = X_random[batch_iter * batch_size_data:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
                y_train = X_random[batch_iter * batch_size_data:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
                t_train = X_random[batch_iter * batch_size_data:, 2].reshape(-1, 1).clone().requires_grad_(True).to(device)
                u_train = X_random[batch_iter * batch_size_data:, 3].reshape(-1, 1).clone().requires_grad_(True).to(device)
                v_train = X_random[batch_iter * batch_size_data:, 4].reshape(-1, 1).clone().requires_grad_(True).to(device)
                p_train = X_random[batch_iter * batch_size_data:, 5].reshape(-1, 1).clone().requires_grad_(True).to(device)
                x_eqa = Eqa_points[batch_iter * batch_size_eqa:, 0].reshape(-1, 1).clone().requires_grad_(True).to(device)
                y_eqa = Eqa_points[batch_iter * batch_size_eqa:, 1].reshape(-1, 1).clone().requires_grad_(True).to(device)
                t_eqa = Eqa_points[batch_iter * batch_size_eqa:, 2].reshape(-1, 1).clone().requires_grad_(True).to(device)
        mse_predict = pinn_net.data_mse(x_train, y_train, t_train, u_train, v_train, p_train)
        mse_equation = pinn_net.equation_mse(x_eqa, y_eqa, t_eqa, lam1=1.0, lam2=0.01)
        # 计算损失函数,不引入压强场的真实值
        loss = mse_predict+mse_equation
        loss.backward()
        optimizer.step()
        with torch.autograd.no_grad():
            # 输出状态
            # if (batch_iter + 1) % 20 == 0:
            #     print("Epoch:", (epoch + 1), "  Bacth_iter:", batch_iter + 1, " Training Loss:",
            #           round(float(loss.data), 8))
            # 每1个epoch保存状态（模型状态,loss,迭代次数）
            if (batch_iter+1) % inner_iter == 0:
                torch.save(pinn_net.state_dict(), filename_save_model)
                loss_all = loss.cpu().data.numpy().reshape(1, 1)
                loss_predict = mse_predict.cpu().data.numpy().reshape(1, 1)
                loss_equation = mse_equation.cpu().data.numpy().reshape(1, 1)
                loss_set = np.concatenate((loss_all, loss_predict, loss_equation), 1)
                losses = np.append(losses, loss_set, 0)
                loss_save = pd.DataFrame(losses)
                loss_save.to_csv(filename_loss, index=False, header=False)
                del loss_save
    scheduler.step()
print("one oK")
torch.save(pinn_net.state_dict(), filename_save_model)
