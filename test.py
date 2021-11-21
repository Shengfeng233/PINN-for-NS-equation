# 分析测试集的表现效果
import numpy as np
import torch
from pinn_model import *

x, y, t, u, v, p, N, T = read_data(filename_data)
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
pinn_net = PINN_Net(layer_mat)
pinn_net.load_state_dict(torch.load(filename_load_model, map_location=device))
mse = nn.MSELoss()


# 可测试部分数据点（如固定时刻），也可测试全流场数据点
def test_part_data(x, y, t, u, v, p):
    x_test = x.clone().requires_grad_(True).to(device)
    y_test = y.clone().requires_grad_(True).to(device)
    t_test = t.clone().requires_grad_(True).to(device)
    u_test = u.clone().requires_grad_(True).to(device)
    v_test = v.clone().requires_grad_(True).to(device)
    p_test = p.clone().requires_grad_(True).to(device)
    zeros = np.zeros((x_test.size(0), 1))
    test_zeros = torch.tensor(zeros, requires_grad= True, dtype=torch.float32).to(device)
    u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_inverse(x_test, y_test, t_test, pinn_net)
    mse_u = mse(u_predict, u_test)
    mse_v = mse(v_predict, v_test)
    mse_p = mse(p_predict, p_test)
    mse_prediction = mse_u + mse_v + mse_p
    mse_equation = mse(f_equation_x, test_zeros) + mse(f_equation_y, test_zeros)
    mse_all = mse_prediction + mse_equation
    return mse_all, mse_prediction, mse_equation, mse_u, mse_v, mse_p


N_test = 5000
index_test = np.random.choice(len(x), N_test, replace=False)
x_test = x[index_test, :]
y_test = y[index_test, :]
t_test = t[index_test, :]
u_test = u[index_test, :]
v_test = v[index_test, :]
p_test = p[index_test, :]
mse_all, mse_prediction, mse_equation, mse_u, mse_v, mse_p = test_part_data(x_test, y_test, t_test, u_test, v_test, p_test)
print("Total_loss:", mse_all.data.numpy())
print("Prediction_loss:", mse_prediction.data.numpy())
print("Equation_loss:", mse_equation.data.numpy())