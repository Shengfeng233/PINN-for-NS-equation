import scipy.io
import numpy as np
import torch
import torch.nn as nn
from pyDOE import lhs

# 定义PINN网络模块，包括数据读取函数，参数初始化
# 正问题和反问题的偏差和求导函数
# 全局参数
filename_load_model = './NS_model_train.pt'
filename_save_model = './NS_model_train.pt'
filename_data = './cylinder_nektar_wake.mat'
filename_loss = './loss.csv'
# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
    print("wrong device")


# 读取原始数据,并转化为x,y,t--u,v,p(N*T,1),返回值为Tensor类型
def read_data(filename):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T

    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, N, T, feature_mat


def read_data_del_hole(filename, hole):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T

    # 对数据进行矩形挖洞
    dig_x = hole[0, :].reshape(-1, 1)
    dig_y = hole[1, :].reshape(-1, 1)

    X_del_index = np.array(np.where(
        (X_star[:, 0:1] >= dig_x[0, 0]) & (X_star[:, 0:1] <= dig_x[1, 0]) & (X_star[:, 1:2] >= dig_y[0, 0]) & (
                    X_star[:, 1:2] <= dig_y[1, 0]))[0])
    X_del_index = X_del_index.reshape(-1, 1)
    X_star_del = np.delete(X_star, X_del_index, 0)
    U_star_del = np.delete(U_star, X_del_index, 0)
    P_star_del = np.delete(P_star, X_del_index, 0)
    # 读取坐标点数N和时间步数T
    N = X_star_del.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star_del[:, 0:1], (1, T))
    YY = np.tile(X_star_del[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star_del[:, 0, :]
    VV = U_star_del[:, 1, :]
    PP = P_star_del
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


def read_data_hole(filename, hole):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T
    T = T_star.shape[0]
    # 对数据进行矩形挖洞
    dig_x = hole[0, :].reshape(-1, 1)
    dig_y = hole[1, :].reshape(-1, 1)

    X_del_index = np.array(np.where(
        (X_star[:, 0:1] >= dig_x[0, 0]) & (X_star[:, 0:1] <= dig_x[1, 0]) & (X_star[:, 1:2] >= dig_y[0, 0]) & (
                    X_star[:, 1:2] <= dig_y[1, 0]))[0])
    X_del_index = X_del_index.reshape(-1, 1)
    X_star_del = X_star[X_del_index, :].reshape(-1, 2)
    U_star_del = U_star[X_del_index, :, :].reshape(-1, 2, T)
    P_star_del = P_star[X_del_index, :].reshape(-1, T)
    # 读取坐标点数N和时间步数T
    N = X_star_del.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star_del[:, 0:1], (1, T))
    YY = np.tile(X_star_del[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star_del[:, 0, :]
    VV = U_star_del[:, 1, :]
    PP = P_star_del
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


def read_data_portion(filename, portion):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T

    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    index_arr_x = np.linspace(0, len(x_unique) - 1, int(len(x_unique) * portion)).astype(int).reshape(-1, 1)
    index_arr_y = np.linspace(0, len(y_unique) - 1, int(len(y_unique) * portion)).astype(int).reshape(-1, 1)
    x_select = x_unique[index_arr_x].reshape(-1, 1)
    y_select = y_unique[index_arr_y].reshape(-1, 1)
    del x_unique, y_unique, index_arr_x, index_arr_y
    index_x = np.empty((0, 1), dtype=int)
    index_y = np.empty((0, 1), dtype=int)
    for select_1 in x_select:
        index_x = np.append(index_x, np.where(x == select_1)[0].reshape(-1, 1), 0)
    for select_2 in y_select:
        index_y = np.append(index_y, np.where(y == select_2)[0].reshape(-1, 1), 0)
    index_all = np.intersect1d(index_x, index_y, assume_unique=False, return_indices=False).reshape(-1, 1)
    x = x[index_all].reshape(-1, 1)
    y = y[index_all].reshape(-1, 1)
    t = t[index_all].reshape(-1, 1)
    u = u[index_all].reshape(-1, 1)
    v = v[index_all].reshape(-1, 1)
    p = p[index_all].reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


def read_data_part_time(filename, portion):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T

    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    t_unique = np.unique(t).reshape(-1, 1)
    index_arr_t = np.linspace(0, len(t_unique) - 1, int(len(t_unique) * portion)).astype(int).reshape(-1, 1)
    t_select = t_unique[index_arr_t].reshape(-1, 1)
    del t_unique, index_arr_t
    index_t = np.empty((0, 1), dtype=int)
    for select_1 in t_select:
        index_t = np.append(index_t, np.where(t == select_1)[0].reshape(-1, 1), 0)
    x = x[index_t].reshape(-1, 1)
    y = y[index_t].reshape(-1, 1)
    t = t[index_t].reshape(-1, 1)
    u = u[index_t].reshape(-1, 1)
    v = v[index_t].reshape(-1, 1)
    p = p[index_t].reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


def read_data_portion_time_expor(filename, portion):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T

    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T))
    YY = np.tile(X_star[:, 1:2], (1, T))
    TT = np.tile(T_star, (1, N)).T
    UU = U_star[:, 0, :]
    VV = U_star[:, 1, :]
    PP = P_star
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    temp = np.concatenate((x, y, t, u, v, p), 1)
    feature_mat = np.empty((2, 6))
    feature_mat[0, :] = np.max(temp, 0)
    feature_mat[1, :] = np.min(temp, 0)
    t_unique = np.sort(np.unique(t).reshape(-1, 1))
    portion_time = int(len(t_unique) * portion)
    t_select = t_unique[0:portion_time, 0].reshape(-1, 1)
    del t_unique
    index_t = np.empty((0, 1), dtype=int)
    for select_1 in t_select:
        index_t = np.append(index_t, np.where(t == select_1)[0].reshape(-1, 1), 0)
    x = x[index_t].reshape(-1, 1)
    y = y[index_t].reshape(-1, 1)
    t = t[index_t].reshape(-1, 1)
    u = u[index_t].reshape(-1, 1)
    v = v[index_t].reshape(-1, 1)
    p = p[index_t].reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    feature_mat = torch.tensor(feature_mat, dtype=torch.float32)
    return x, y, t, u, v, p, feature_mat


# 定义网络结构,由layer列表指定网络层数和神经元数
class PINN_Net(nn.Module):
    def __init__(self, layer_mat):
        super(PINN_Net, self).__init__()
        self.layer_num = len(layer_mat) - 1
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            # nn.init.kaiming_normal()
            self.base.add_module(str(i) + "Act", nn.Tanh())
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.lam1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.lam2 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.Initial_param()

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    # 对参数进行初始化
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)

    # 类内方法：求数据点的loss
    def data_mse(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        p_predict = predict_out[:, 1].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # 类内方法：求数据点的loss(不含压力数据)
    def data_mse_without_p(self, x, y, t, u, v):
        predict_out = self.forward(x, y, t)
        psi = predict_out[:, 0].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # 类内方法：求方程点的loss
    def equation_mse(self, x, y, t, lam1, lam2):
        # 正问题,需要用户自行提供系统的参数值，默认为1&0.01
        predict_out = self.forward(x, y, t)
        # 获得预测的输出psi,p
        psi = predict_out[:, 0].reshape(-1, 1)
        p = predict_out[:, 1].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # 计算偏微分方程的残差
        f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros)
        return mse_equation


# 生成矩形域方程点
def generate_eqp_rect(low_bound, up_bound, dimension, points):
    eqa_xyzt = low_bound + (up_bound - low_bound) * lhs(dimension,points)
    per = np.random.permutation(eqa_xyzt.shape[0])
    new_xyzt = eqa_xyzt[per, :]
    Eqa_points = torch.from_numpy(new_xyzt).float()
    return Eqa_points


# 定义偏微分方程（的偏差）inverse为反问题
def f_equation_inverse(x, y, t, pinn_example):
    lam1 = pinn_example.lam1
    lam2 = pinn_example.lam2
    predict_out = pinn_example.forward(x, y, t)
    # 获得预测的输出psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 计算偏微分方程的残差
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, p, f_equation_x, f_equation_y


def f_equation_identification(x, y, t, pinn_example, lam1=1.0, lam2=0.01):
    # 正问题,需要用户自行提供系统的参数值，默认为1&0.01
    predict_out = pinn_example.forward(x, y, t)
    # 获得预测的输出psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 计算偏微分方程的残差
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, p, f_equation_x, f_equation_y


def f_equation_uv_identification(x, y, t, pinn_example, lam1=1.0, lam2=0.01):
    # 正问题,需要用户自行提供系统的参数值，默认为1&0.01
    predict_out = pinn_example.forward(x, y, t)
    # 获得预测的输出psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 计算偏微分方程的残差
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, p, f_equation_x, f_equation_y


def f_equation_inverse_simple_norm(x, y, t, feature_mat, pinn_example):
    lam1 = pinn_example.lam1
    lam2 = pinn_example.lam2
    predict_out = pinn_example.forward(x, y, t)
    # 获得预测的输出psi,p
    u = predict_out[:, 0].reshape(-1, 1)
    v = predict_out[:, 1].reshape(-1, 1)
    p = predict_out[:, 2].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 由于对数据进行了归一化,需要对方程进行修改
    # 计算偏微分方程的残差,包含连续性方程和动量方程等三个方程
    [L1, L2, t0, U1, U2, p0] = feature_mat[0, :]
    f_equation_c = u_x * U1 / L1 + v_y * U2 / L2
    f_equation_x = u_t * U1 / t0 + u * u_x * U1 * U1 / L1 + v * u_y * U1 * U2 / L2 + lam1 * p_x * p0 / L1 - lam2 * (
                u_xx * U1 / (L1 * L1) + u_yy * U1 / (L2 * L2))
    f_equation_y = v_t * U2 / t0 + u * v_x * U1 * U2 / L1 + v * v_y * U2 * U2 / L2 + lam1 * p_y * p0 / L2 - lam2 * (
                v_xx * U2 / (L1 * L1) + v_yy * U2 / (L2 * L2))
    return u, v, p, f_equation_c, f_equation_x, f_equation_y


def shuffle_data(x, y, t, u, v, p):
    X_total = torch.cat([x, y, t, u, v, p], 1)
    X_total_arr = X_total.data.numpy()
    np.random.shuffle(X_total_arr)
    X_total_random = torch.tensor(X_total_arr)
    return X_total_random


def simple_norm(x, y, t, u, v, p, feature_mat):
    x = x / feature_mat[0, 0]
    y = y / feature_mat[0, 1]
    t = t / feature_mat[0, 2]
    u = u / feature_mat[0, 3]
    v = v / feature_mat[0, 4]
    p = p / feature_mat[0, 5]
    return x, y, t, u, v, p, feature_mat
