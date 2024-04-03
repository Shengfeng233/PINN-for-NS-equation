import scipy
from pinn_model import *


def load_valid_points(filename):
    # for each process, load data points within the extended boundary
    # 每一个进程读取相应区域延拓边界内的数据点
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


def compute_L2_norm(filename_raw_data, pinn_net, device, norm_status="no_norm"):
    # 预测值
    x_raw, y_raw, t_raw, u_raw, v_raw, p_raw, low_bound, up_bound = load_valid_points(filename_raw_data)
    x_pre = x_raw.clone().detach().requires_grad_(True).to(device)
    y_pre = y_raw.clone().detach().requires_grad_(True).to(device)
    t_pre = t_raw.clone().detach().requires_grad_(True).to(device)
    if norm_status == "no_norm":
        u_pre, v_pre, p_pre = pinn_net.predict(x_pre, y_pre, t_pre)
    else:
        u_pre, v_pre, p_pre = pinn_net.predict_inner_norm(x_pre, y_pre, t_pre)
    u_raw_mat = u_raw.numpy()
    v_raw_mat = v_raw.numpy()
    p_raw_mat = p_raw.numpy()
    u_pre_mat = u_pre.cpu().detach().numpy()
    v_pre_mat = v_pre.cpu().detach().numpy()
    p_pre_mat = p_pre.cpu().detach().numpy()
    # 处理数据
    L2_u = (np.linalg.norm(u_pre_mat - u_raw_mat) / np.linalg.norm(u_raw_mat)).reshape(-1, 1)
    L2_v = (np.linalg.norm(v_pre_mat - v_raw_mat) / np.linalg.norm(v_raw_mat)).reshape(-1, 1)
    L2_p = (np.linalg.norm(p_pre_mat - p_raw_mat) / np.linalg.norm(p_raw_mat)).reshape(-1, 1)
    return L2_u, L2_v, L2_p


