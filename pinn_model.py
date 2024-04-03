"""
PINN+incompressible NS equation
2-dimensional unsteady
PINN model +LOSS function
PINN融合不可压缩NS方程
二维非定常流动
PINN模型 + LOSS函数
"""
import os
import numpy as np
import torch
import torch.nn as nn
from pyDOE import lhs


# Define network structure, specified by a list of layers indicating the number of layers and neurons
# 定义网络结构,由layer列表指定网络层数和神经元数
class PINN_Net(nn.Module):
    def __init__(self, layer_mat, mean_value, std_value, device):
        super(PINN_Net, self).__init__()
        self.X_mean = torch.from_numpy(mean_value.astype(np.float32)).to(device)
        self.X_std = torch.from_numpy(std_value.astype(np.float32)).to(device)
        self.device = device
        self.layer_num = len(layer_mat) - 1
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            self.base.add_module(str(i) + "Act", nn.Tanh())
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.Initial_param()

    # 0-1 norm of input variable
    # 对输入变量进行0-1归一化
    def inner_norm(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        return X_norm

    def forward_inner_norm(self, x, y, t):
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        X_norm = self.inner_norm(X)
        predict = self.base(X_norm)
        return predict

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    # initialize
    # 对参数进行初始化
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('linear.weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('linear.bias'):
                nn.init.zeros_(param)

    # derive loss for data
    # 类内方法：求数据点的loss
    def data_mse(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # derive loss for data without pressure
    # 类内方法：求数据点的loss(不含压力数据)
    def data_mse_without_p(self, x, y, t, u, v):
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # predict
    # 类内方法：预测
    def predict(self, x, y, t):
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        return u_predict, v_predict, p_predict

    # derive loss for equation
    def equation_mse_dimensionless(self, x, y, t, Re):
        predict_out = self.forward(x, y, t)
        # 获得预测的输出u,v,p
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation

    def data_mse_inner_norm(self, x, y, t, u, v, p):
        predict_out = self.forward_inner_norm(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # derive loss for data without pressure
    # 类内方法：求数据点的loss(不含压力数据)
    def data_mse_without_p_inner_norm(self, x, y, t, u, v):
        predict_out = self.forward_inner_norm(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # predict
    # 类内方法：预测
    def predict_inner_norm(self, x, y, t):
        predict_out = self.forward_inner_norm(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        return u_predict, v_predict, p_predict

    # derive loss for equation
    def equation_mse_dimensionless_inner_norm(self, x, y, t, Re):
        predict_out = self.forward_inner_norm(x, y, t)
        # 获得预测的输出u,v,w,p,k,epsilon
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation
