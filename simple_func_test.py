import numpy as np

from pinn_model import *
import pandas
import matplotlib.pyplot as plt

filename_data = './cylinder_nektar_wake.mat'
# #the .2f specifys the number of trailing values after the decimal you want
# final = "{:.2f}".format(value)
#
# x, y, t, u, v, p, feature_mat = read_data_all(filename_data)
# [L1, L2, t0, U1, U2, p0] = feature_mat[0,:]
# a = U1*U2
# print(final)


# file_loss = pd.read_csv(filename_loss, header=None)
# loss = file_loss.values
# x = np.linspace(1,loss.shape[0],loss.shape[0]).reshape(-1,1)
# total_loss = loss[:, 0].reshape(-1,1)
# predict_loss = loss[:, 1].reshape(-1,1)
# equation_loss = loss[:, 2].reshape(-1,1)
# plt.title('LOSS')  # 折线图标题
# plt.xlabel('iter')  # x轴标题
# plt.ylabel('loss_value')  # y轴标题
# plt.semilogy(x, total_loss, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
# plt.semilogy(x, predict_loss, marker='o', markersize=3)
# plt.semilogy(x, equation_loss, marker='o',markersize=3)
# plt.legend(['total_loss', 'predict_loss', 'equation_loss'])
# plt.show()
# a = np.linspace(1,100,100).astype(int).reshape(-1,1)
# b = np.linspace(1,100,50).astype(int).reshape(-1,1)
# index_time = np.where(a <= 6)[0]
# c = a[b-1].squeeze()
portion = 0.75
x, y, t, u, v, p, feature_mat = read_data_portion_time_expor(filename_data, portion)
print("ok")