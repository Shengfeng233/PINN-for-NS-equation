from utilities import *
import time
from argparse import Namespace
from validation_func import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

train_config = Namespace(
    write_path='./write',
    loss_path='./write/loss.csv',
    evaluate_path='./write/RL2.csv',
    data_path='./cylinder_Re3900_36points_100snaps.mat',
    real_path='./cylinder_Re3900_ke_all_100snaps.mat',
    Re=3900,
    dimension=2 + 1,
    outer_epoch=5000,
    inner_epochs=1,
    save_interval=100,
    number_eqa=1000000,
    inner_Norm="with_norm",
    optimizer='adam',
    scheduler='exp',
    batch_size=10000,
    learning_rate=1e-3,
    hidden_layers=10,
    layer_neurons=32,
    weight_of_data=1,
    weight_of_eqa=1,
    debug_key=1,
)


def build_optimizer(network, optimizer_name, scheduler_name, learning_rate):
    # default 默认优化器
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    # 超参数搜索优化器
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    if scheduler_name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif scheduler_name == "fix":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    return optimizer, scheduler


def validation(pinn_net, real_path, inner_Norm, filename_RL2):
    RL2_u, RL2_v, RL2_p = compute_L2_norm(real_path, pinn_net, device, norm_status=inner_Norm)
    RL2_u_value = RL2_u.reshape(1, 1)
    RL2_v_value = RL2_v.reshape(1, 1)
    RL2_p_value = RL2_p.reshape(1, 1)
    RL2_set = np.concatenate((RL2_u_value, RL2_v_value, RL2_p_value), 1).reshape(1, -1)
    ReL2_save = pd.DataFrame(RL2_set)
    ReL2_save.to_csv(filename_RL2, index=False, header=False, mode='a')
    return RL2_u_value, RL2_v_value, RL2_p_value


def train():
    # important parameters
    # 重要参数
    data_path = train_config.data_path  # 训练数据
    Re = train_config.Re  # 雷诺数 Reynolds number
    dimension = train_config.dimension
    real_path = train_config.real_path
    write_path = train_config.write_path
    evaluate_path = train_config.evaluate_path
    loss_path = train_config.loss_path
    inner_epochs = train_config.inner_epochs
    save_interval = train_config.save_interval  # 保存模型周期
    outer_epochs = train_config.outer_epoch  # 训练周期数 epoch-outer epoch
    number_eqa = train_config.number_eqa
    debug_key = train_config.debug_key
    inner_Norm = train_config.inner_Norm
    # 搜索超参数
    layer_mat = [dimension] + train_config.hidden_layers * [train_config.layer_neurons] + [3]
    learning_rate = train_config.learning_rate  # 学习率 learning rate
    batch_size = train_config.batch_size
    weight_of_data = train_config.weight_of_data
    weight_of_eqa = train_config.weight_of_eqa
    optimizer_name = train_config.optimizer
    scheduler_name = train_config.scheduler

    # pretraining-loading data
    # loading data points
    # loading collocation points
    # only load once
    # 训练前加载工作(仅加载一次)
    # 加载数据点，加载方程点
    True_dataset_batches, Eqa_points_batches, iter_num, low_bound, up_bound = pre_train_loading(
        data_path,
        dimension,
        number_eqa,
        batch_size, )
    data_mean, data_std = load_data_feature(data_path)
    pinn_net = PINN_Net(layer_mat, data_mean, data_std, device)
    pinn_net = pinn_net.to(device)

    # 优化器和学习率衰减设置- optimizer and learning rate schedule
    optimizer_all, scheduler_all = build_optimizer(pinn_net, optimizer_name, scheduler_name, learning_rate)

    start = time.time()
    if not os.path.exists(write_path):
        # 创建文件夹 create file for recording
        os.mkdir(write_path)
    # 训练主循环 main loop
    for EPOCH in range(outer_epochs):

        # 对数据点，方程点进行同时训练 train----每一个iter中训练全部数据
        # 不设置内嵌归一化层 no inner normalization
        if inner_Norm == "no_norm":
            loss_sum, loss_data, loss_eqa = train_data_whole(inner_epochs, pinn_net, optimizer_all, scheduler_all,
                                                             iter_num, True_dataset_batches, Eqa_points_batches,
                                                             Re, weight_of_data, weight_of_eqa, EPOCH, debug_key,
                                                             device)
        # 设置内嵌归一化层 with inner normalization
        else:
            loss_sum, loss_data, loss_eqa = train_data_whole_inner_norm(inner_epochs, pinn_net, optimizer_all,
                                                                        scheduler_all,
                                                                        iter_num, True_dataset_batches,
                                                                        Eqa_points_batches, Re, weight_of_data,
                                                                        weight_of_eqa, EPOCH,
                                                                        debug_key, device)

        # loss记录 record loss
        loss_set = record_loss_local(loss_sum, loss_data, loss_eqa, loss_path)  # 记录子网络loss

        # 每隔固定Epoch保存模型 save model at every save_interval epoch
        # 每隔固定Epoch评估模型 evaluate model at every save_interval epoch
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        if ((EPOCH + 1) % save_interval == 0) | (EPOCH == 0):
            dir_name = write_path + '/step' + str(EPOCH + 1)
            os.makedirs(dir_name, exist_ok=True)
            torch.save(pinn_net.state_dict(), dir_name + '/NS_model_train.pt')
            print(f'Model saved at step {EPOCH + 1}.')
            valid_u, valid_v, valid_p = validation(pinn_net, real_path, inner_Norm, evaluate_path)
            print(f'Model evaluated at step {EPOCH + 1}.')

    end = time.time()
    print("Time used: ", end - start)
    return


if __name__ == '__main__':
    train()
