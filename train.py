from pinn_model import *
import time
import pandas as pd
import os



# 训练代码主体
x, y, t, u, v, p, N, T = read_data(filename_data)
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
X_random = shuffle_data(x, y, t, u, v, p)
# 创建PINN模型实例，并将实例分配至对应设备
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)
# 损失函数和优化器
mse = torch.nn.MSELoss()
losses = []
if os.path.exists(filename_save_model):
    pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
if os.path.exists(filename_loss):
    loss_read = pd.read_csv('loss.csv', header=None)
    losses = loss_read.values
    losses = list(losses)
optimizer = torch.optim.Adam(pinn_net.parameters(), lr=0.00001)
epochs = 3
start_time = time.time()

# 选取batch size 此处也可使用data_loader
batch_size = 500
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
        u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_inverse(batch_t_x, batch_t_y, batch_t_t,
                                                                                         pinn_net)

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
                losses.append(loss.data.numpy().reshape(1, 1))
                print("Epoch:", (epoch+1), "  Bacth_iter:", batch_iter + 1, " Training Loss:", round(float(loss.data), 8),'lam1 = ', pinn_net.lam1.data, "lam2 = ", pinn_net.lam2.data)
            # 每1个epoch保存状态（模型状态,loss,迭代次数）
            if (batch_iter + 1) % inner_iter == 0:
                torch.save(pinn_net.state_dict(), filename_save_model)
                loss_save = pd.DataFrame(losses)
                loss_save.to_csv(filename_loss, index=False, header=False)
print("one oK")
torch.save(pinn_net.state_dict(), filename_save_model)