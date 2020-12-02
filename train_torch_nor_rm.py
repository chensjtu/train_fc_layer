import model
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
import numpy as np
import time
import copy
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import glob
import argparse
# from sklearn import preprocessing


class ARKit_ARCore_Dataset(Dataset):
    def __init__(self, df_data, transform=None):
        self.df_data = df_data
        self.transform = transform

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        # data = self.df_data.iloc[idx, :]
        # if self.transform:
        #     data = self.transform(data)
        return self.df_data.iloc[idx, 0:62].values.astype('float32'), self.df_data.iloc[idx, 62:].values.astype(
            'float32')


def cal_mse(out, label):
    return np.power((out-label), 2)


def cal_blendshape_loss(outputs, labels):
    for num, i in enumerate(outputs):
        if num == 0:
            outputs_array = i
        else:
            outputs_array = np.vstack((outputs_array, i))
    for num, i in enumerate(labels):
        if num == 0:
            labels_array = i
        else:
            labels_array = np.vstack((labels_array, i))
    print(outputs_array.shape, labels_array.shape, type(outputs_array), type(labels_array))
    # a[:, 1] means the first lie
    loss = []
    for i in range(52):
        loss.append(np.mean(cal_mse(outputs_array[:,i], labels_array[:,i])))
    return loss


def train_model(model, criterion, optimizer, scheduler, train_data, val_data, device, num_epochs=1000):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 1
    all_train_loss = []
    all_val_loss = []
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        running_loss = 0.0
        # Each epoch has a training and validation phase
        if epoch < 0 or (epoch % 300):
            pass
        else:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(checkpoint, './checkpoint/ckpt_best_%s.pth' % (str(epoch)))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                for inputs, labels in train_data:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(indexed_train)
                all_train_loss.append(epoch_loss)
                if epoch % 100 == 0:
                    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                    logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
            else:
                model.eval()  # Set model to evaluate mode
                for inputs, labels in val_data:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(indexed_valid)
                all_val_loss.append(epoch_loss)
                if epoch % 100 == 0:
                    # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                    # logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # deep copy the model
            if phase == 'val' and epoch_loss < best_acc:
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:5f}'.format(best_acc))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val loss: {:5f}'.format(best_acc))

    outputs_np = []
    labels_np = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_np.append(outputs.cpu().numpy())
            labels_np.append(labels.cpu().numpy())
        mse_loss = cal_blendshape_loss(outputs_np, labels_np)
        np.savetxt("./checkpoint/mse_loss.txt", np.array(mse_loss))

    plt.figure()
    # draw learning rate
    plt.plot(all_train_loss, c="r", label="train_loss")
    plt.plot(all_val_loss, c="g", label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train_loss")
    plt.savefig("./checkpoint/train_loss.png")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# def sep(df):
#     # mean_var = []
#     scaler_d = preprocessing.StandardScaler()
#     df_x, x_mean, x_var = nor(df.iloc[:, 1:469])
#     df_y, y_mean, y_var = nor(df.iloc[:, 470:938])
#     df_z, z_mean, z_var = nor(df.iloc[:, 939:1407])
#     # 469, 938, 1407 is the degree.
#     degree = np.array(df.iloc[:, [469, 938, 1407]].values.astype('float32'))
#     # judge the degree [0, 180], [180, 360]-360,
#     scaler_d.fit(degree)
#     labels = np.array(df.iloc[:, 1408:1460].values.astype('float32'))
#     # mean_var.append(x_mean, y_mean, z_mean, scaler_d.mean_, x_var, y_var, z_var, scaler_d.var_)
#     return np.hstack((df_x, df_y, df_z, scaler_d.fit_transform(degree), labels)), \
#            np.hstack((x_mean, y_mean, z_mean, scaler_d.mean_, x_var, y_var, z_var, scaler_d.var_))


def nor(df):
    ios_df = df.iloc[:, 62:]
    pc_df = df.iloc[:, 0:62]
    pc_df_std = np.std(pc_df)
    pc_df_mean = np.mean(pc_df)
    np.save("./pc_df_std.npy", pc_df_std)
    np.save("./pc_df_mean.npy", pc_df_mean)
    print(pc_df_std, pc_df_mean)
    pc_dfstd = pc_df.copy()
    pc_dfstd.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    return pc_dfstd, ios_df


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='train.log', level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description='train blend shapes to pc')
    parser.add_argument('--local', type=str, default="./data", help='data location')
    parser.add_argument('--nor', type=bool, default=False, help='normalization')
    parser.add_argument('--pre_train', type=str, default="./model_statedic.pt", help='pt position')
    parser.add_argument('--ctu', type=bool, default=False, help='continue choose true')
    parser.add_argument('--rm', type=bool, default=True, help='remove ten useless labels')
    args = parser.parse_args()
    print(args)

    dataset_list = []
    file = glob.glob(r"./data/*.csv")
    for f in file: dataset_list.append(pd.read_csv(f, header=None))
    dataset = pd.concat(dataset_list)
    # assert not np.any(dataset.isnull())
    assert dataset.shape[1] == 114
    logging.info(dataset.shape)

    if args.nor:
        pc_data, ios_data = nor(dataset)
        dataset = pd.concat([pc_data, ios_data], ignore_index=True, join="outer", axis=1)
        print(dataset.shape, 'normalize is applied')

    # nums_worker using the number of CPU xiancheng.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.rm:
        net = model.Pro_net(in_dim=62, n_hidden_1=128, n_hidden_2=256, n_hidden_3=512, n_hidden_4=1024,
                            n_hidden_5=512, n_hidden_6=256, n_hidden_7=128, out_dim=42)
        need_remove_list = [62, 63, 70, 71, 80, 81, 95, 96, 105, 106] # means the bad loss position + 62
        dataset = dataset.iloc[:, [i for i in range(dataset.shape[1]) if i not in need_remove_list]]
        dataset.reset_index(drop=True, inplace=True)
        dataset.columns = [x for x in range(0, dataset.shape[1])]
    else:
        net = model.Pro_net(in_dim=62, n_hidden_1=128, n_hidden_2=256, n_hidden_3=512, n_hidden_4=1024,
                            n_hidden_5=512, n_hidden_6=256, n_hidden_7=128, out_dim=52)

    arr_indexes = [i for i in range(len(dataset))]
    # len(data) = data.shape[0]
    random.seed(0)
    random.shuffle(arr_indexes)
    indexed_train = arr_indexes[:int(len(arr_indexes) * 3 / 4)]
    indexed_valid = arr_indexes[int(len(arr_indexes) * 3 / 4):]
    df_train = dataset.iloc[indexed_train, :]
    df_valid = dataset.iloc[indexed_valid, :]

    train_data = DataLoader(ARKit_ARCore_Dataset(df_train), batch_size=2048, shuffle=True, pin_memory=True,
                            num_workers=16)
    logging.info("train")
    val_data = DataLoader(ARKit_ARCore_Dataset(df_valid), batch_size=2048, shuffle=True, pin_memory=True,
                          num_workers=16)
    logging.info("val")

    if torch.cuda.is_available():
        net.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
    else:
        net.cpu()

    loss_fn = torch.nn.MSELoss()
    # 3e-4 is the best learning rate for adam, hands down.
    # Amsgrad for the ICLR best paper may induce faster.
    optimizer = optim.Adam(net.parameters(), lr=3e-4, amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    if args.ctu:
        path_checkpoint = "./checkpoint/ckpt_best_1.pth"  # 断点路径
        checkpoint = torch.load(args.pre_train)  # 加载断点

        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch


    new_model = train_model(net, criterion=loss_fn, optimizer=optimizer, scheduler=scheduler,
                            train_data=train_data, val_data=val_data, device=device, num_epochs=1500)
    if not os.path.isdir("./checkpoint"):
        os.mkdir("./checkpoint")
    torch.save(new_model.module.state_dict(), "./checkpoint/model_best_statedic.pt")
    torch.save(new_model, "./checkpoint/model_best.pt")

