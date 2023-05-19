import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader

# data preprocess
import numpy as np
import csv
import os

# utils
from utils import get_device, plot_learning_curve, plot_pred

# 訓練資料
tr_path = 'covid.train.csv'

# 測試資料
# 通常可能要自己切？
tt_path = 'covid.test.csv'

# this is for reproducibility(可復現)
myseed = 42069
torch.backends.cudnn.deteministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # read data from numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            # all the data in this dataset is float(except id and title)
            # data[1:] -> don't want title
            # data[:, 1:] -> don't want id, keep all row, drop column 1
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            # column 0 -> 92
            feats = list(range(93))
        else:
            # why choose this way?
            feats = list(range(40)) + [57, 75]
            pass

        if mode == 'test':
            # testing data 893 x 93
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # mode == 'train' or else
            # training data 2700 x 94

            # select last column
            target = data[:, -1]
            # because last column is target value
            data = data[:, feats]

            # split into training and dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            else:
                indices = []
                pass

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # 沒有normalize的話根本train不動
        # normalize features
        # 0-39 are one-hot states, 40: are features
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


# x = COVID19Dataset(tr_path, mode='test')

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == 'train'),
        drop_last=False,
        num_workers=n_jobs,
        pin_memory=True
    )
    return dataloader


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # network
        # 跟 dense 一樣？
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # mse loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        # TODO: you may implement L1/L2 regularization here
        # L1/L2 regularization is for dealing with overfitting
        # https://www.youtube.com/watch?v=VqKq78PVO9g&ab_channel=codebasics
        return self.criterion(pred, target)


# hyper-parameters

# get the current available device ('cpu' or 'cuda')
device = get_device()
# The trained model will be saved to ./models/
os.makedirs('models', exist_ok=True)
# TODO: Using 40 states & 2 tested_positive features
target_only = False

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 150,               # mini-batch size for dataloader
    # optimization algorithm (optimizer in torch.optim)
    'optimizer': 'SGD',
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.003,                 # learning rate of SGD
        'momentum': 0.9             # momentum for SGD
    },
    # early stopping epochs (the number epochs since your model's last improvement)
    'early_stop': 200,
    'save_path': 'models/model.pth'  # your model will be saved here
}

# 跟train差在哪裡？
# no optimizer, no_grad, epoch loop


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        # why * len(x) ?
        # detach() stop back propagation
        # cpu() move data from gpu to cpu
        # item() get the value of the tensor
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)

    return total_loss


def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']

    # optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0

    while epoch < n_epochs:
        model.train()  # set model to training mode

        # every epoch loop
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()  # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # evaluate our model with dev set
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # this means the model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(
                epoch+1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)

        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


tr_set = prep_dataloader(
    tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(
    tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(
    tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)

model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
