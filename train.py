import numpy as np

from pinn_model import *
import time
import pandas as pd
import os

# 部分时间数据训练（内插&外推）-无数据归一化处理
# 训练代码主体
portion_time = 0.5
x, y, t, u, v, p, feature_mat = read_data_part_time(filename_data, portion_time)
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
X_random = shuffle_data(x, y, t, u, v, p)
view_x = X_random.data.numpy()
# 创建PINN模型实例，并将实例分配至对应设备
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)
# 损失函数和优化器
mse = torch.nn.MSELoss()
# 用以记录各部分损失的列表
losses = np.empty((0, 3), dtype=float)

if os.path.exists(filename_save_model):
    pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
if os.path.exists(filename_loss):
    loss_read = pd.read_csv('loss.csv', header=None)
    losses = loss_read.values
# 优化器和学习率衰减设置
optimizer = torch.optim.Adam(pinn_net.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
epochs = 500

# 选取batch size 此处也可使用data_loader
batch_size = 1000
inner_iter = int(X_random.size(0) / batch_size)

for epoch in range(epochs):
    for batch_iter in range(inner_iter):
        optimizer.zero_grad()
        # 在全集中随机取batch
        x_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 0].view(batch_size, 1)
        y_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 1].view(batch_size, 1)
        t_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 2].view(batch_size, 1)
        u_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 3].view(batch_size, 1)
        v_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 4].view(batch_size, 1)
        p_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 5].view(batch_size, 1)

        # 定于zeros用于计算微分方程误差的MSE
        zeros = np.zeros((batch_size, 1))
        # 将batch从全集中clone出
        batch_t_x = x_train.clone().requires_grad_(True).to(device)
        batch_t_y = y_train.clone().requires_grad_(True).to(device)
        batch_t_t = t_train.clone().requires_grad_(True).to(device)
        batch_t_u = u_train.clone().requires_grad_(True).to(device)
        batch_t_v = v_train.clone().requires_grad_(True).to(device)
        batch_t_p = p_train.clone().requires_grad_(True).to(device)
        batch_t_zeros = torch.from_numpy(zeros).float().requires_grad_(True).to(device)
        # 删除不需要的内存空间
        del x_train, y_train, t_train, u_train, v_train, p_train, zeros

        # 调用f_equation函数进行损失函数各项计算
        u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_identification(batch_t_x, batch_t_y, batch_t_t,
                                                                                         pinn_net,lam1=1.0, lam2=0.01)

        # 计算损失函数
        mse_predict = mse(u_predict, batch_t_u) + mse(v_predict, batch_t_v) + mse(p_predict, batch_t_p)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros)
        loss = mse_predict + mse_equation
        loss.backward()
        optimizer.step()
        with torch.autograd.no_grad():
            # 每200次迭代输出状态
            if (batch_iter + 1) % 200 == 0:
                # 添加loss到losses
                loss_all = loss.cpu().data.numpy().reshape(1, 1)
                loss_predict = mse_predict.cpu().data.numpy().reshape(1, 1)
                loss_equation = mse_equation.cpu().data.numpy().reshape(1, 1)
                loss_set = np.concatenate((loss_all, loss_predict, loss_equation), 1)
                losses = np.append(losses, loss_set, 0)
                print("Epoch:", (epoch+1), "  Bacth_iter:", batch_iter + 1, " Training Loss:", round(float(loss.data), 8))
            # 每1个epoch保存状态（模型状态,loss,迭代次数）
            if (batch_iter + 1) % inner_iter == 0:
                torch.save(pinn_net.state_dict(), filename_save_model)
                loss_save = pd.DataFrame(losses)
                loss_save.to_csv(filename_loss, index=False, header=False)
                del loss_save
    scheduler.step()
print("one oK")
torch.save(pinn_net.state_dict(), filename_save_model)
