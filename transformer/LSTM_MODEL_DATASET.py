import time
import pandas_datareader.data as pdr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import math
import FinanceDataReader as fdr


def metric1(y_pred, y_true):
    perc_y_pred = y_pred.cpu().detach().numpy()
    perc_y_true = y_true.cpu().detach().numpy()

    # mean_absolute_error : 차이의 절댓값을 loss function으로 사용
    mae = mean_absolute_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    print(mae)
    return mae



def metric2(y_pred, y_true):
    perc_y_pred = y_pred.cpu().detach().numpy()
    #print('perc_y_pred :{},perc_y_pred.shape :{}'.format(perc_y_pred,perc_y_pred.shape))
    perc_y_true = y_true.cpu().detach().numpy()
    # mean_absolute_error : 차이의 절댓값을 loss function으로 사용
    mse = mean_squared_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    rmse = math.sqrt(mse)
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # mse = mean_squared_error(y_pred, y_true, multioutput='raw_values')
    return rmse

def metric3(y_pred, y_true):
    perc_y_pred = y_pred.cpu().detach().numpy()
    perc_y_true = y_true.cpu().detach().numpy()
    # mean_absolute_error : 차이의 절댓값을 loss function으로 사용
    mape = mean_absolute_percentage_error(perc_y_true, perc_y_pred, multioutput='raw_values')[0]
    return mape
## t+1 부터 t+5 까지 동시에 예측
## data preperation

## dataset은 i번째 record 값을 출력해주는 역할을 함.
## if input feature is 2 demension, then 인풋으로는 2차원을 주고, 아웃풋으로는 1차원 값 하나를 return 하면 끝나는 역할

## batch를 만들어주는것이 pytorch의 data loader임, random sample도 만들어줌
## data loader는 파이토치에 구현되어있음




class CV_Data_Spliter:
    def __init__(self, symbol, data_start, data_end,n_splits,gap=0):
        self.symbol = symbol
        self.n_splits = n_splits
        # self.start = datetime.datetime(*data_start)
        # self.end = datetime.datetime(*data_end)
        self.start = data_start
        self.end = data_end

        # self.data = pdr.DataReader(self.symbol, 'yahoo', self.start, self.end)
        self.data = fdr.DataReader(self.symbol, self.start, self.end)

        self.chart_data = self.data
        self.test_size = len(self.data)//10-1
        self.gap = gap
        print(self.data.isna().sum())

        self.tscv = TimeSeriesSplit(gap=self.gap, max_train_size=None, n_splits=self.n_splits, test_size=self.test_size)

    def ts_cv_List(self):
        list = []
        for train_index, test_index in self.tscv.split(self.data):
            X_train, X_test = self.data.iloc[train_index, :], self.data.iloc[test_index, :]
            list.append((X_train, X_test))
        return list

    def test_size(self):
        return self.test_size

    def entire_data(self):
        return self.chart_data

    def __len__(self):
        return self.n_splits

    def __getitem__(self, item):
        # data = self.data['Close']  ## time series data를 그대로 가져가기 위해서는 괄호를 한번 넣어야한다.
        # self.chart_data = self.data['Close']
        # data_max, data_min = max(data), min(data)
        # normal = (data - data_min) / (data_max - data_min)
        # self.normed_data = normal
        datalist = self.ts_cv_List()
        return datalist[item]


class CV_train_Spliter:
    def __init__(self, data, symbol, test_size, gap=0):
        self.symbol = symbol
        self.data = data
        self.test_size = test_size
        self.gap = gap
        print(self.data.isna().sum())
        self.tscv = TimeSeriesSplit(gap=self.gap, max_train_size=None, n_splits=2, test_size=self.test_size)

    def ts_cv_List(self):
        list= []
        for train_index, test_index in self.tscv.split(self.data):
            X_train, X_test = self.data.iloc[train_index, :], self.data.iloc[test_index, :]
            list.append((X_train, X_test))
        return list

    def __getitem__(self, item):
        datalist= self.ts_cv_List()
        return datalist[item]


class StockDatasetCV(Dataset):

    def __init__(self, data, x_frames, y_frames):
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.data = data
        print(self.data.isna().sum())

    ## 데이터셋에 len() 을 사용하기 위해 만들어주는것 (dataloader에서 batch를 만들때 이용됨)
    def __len__(self):
        return len(self.data) - (self.x_frames + self.y_frames) + 1

    ## a[:]와 같은 indexing 을 위해 getinem 을 만듬
    ## custom dataset이 list가 아님에도 그 데이터셋의 i번째의 x,y를 출력해줌
    def __getitem__(self, idx):
        idx += self.x_frames
        data = pd.DataFrame(self.data).iloc[idx - self.x_frames:idx + self.y_frames]
        #data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] ## 컬럼순서맞추기 위해 한것
        data = data['Close']
        ## log nomalization
        # data = data.apply(lambda x: np.log(x + 1) - np.log(x[self.x_frames - 1] + 1))
        ## min max normalization
        min_data, max_data = min(data), max(data)
        normed_data = (data-min(data))/(max(data)-min(data))
        data = normed_data.values ## (data.frame >> numpy array) convert >> 나중에 dataloader가 취합해줌
        ## x와 y기준으로 split
        X = data[:self.x_frames]
        y = data[self.x_frames:]

        return X, y, min_data, max_data



# class StockDatasetCV(Dataset):
#
#     def __init__(self,data , x_frames, y_frames):
#         self.x_frames = x_frames
#         self.y_frames = y_frames
#         self.data = data
#         print(self.data.isna().sum())
#
#     ## 데이터셋에 len() 을 사용하기 위해 만들어주는것 (dataloader에서 batch를 만들때 이용됨)
#     def __len__(self):
#         return len(self.data) - (self.x_frames + self.y_frames) + 1
#
#     ## a[:]와 같은 indexing 을 위해 getinem 을 만듬
#     ## custom dataset이 list가 아님에도 그 데이터셋의 i번째의 x,y를 출력해줌
#     def __getitem__(self, idx):
#         idx += self.x_frames
#         data = pd.DataFrame(self.data).iloc[idx - self.x_frames:idx + self.y_frames]
#         #data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] ## 컬럼순서맞추기 위해 한것
#         #data = data[['Close']]
#         ## log nomalization
#         # data = data.apply(lambda x: np.log(x + 1) - np.log(x[self.x_frames - 1] + 1))
#         data = data.values ## (data.frame >> numpy array) convert >> 나중에 dataloader가 취합해줌
#         ## x와 y 기준으로 split
#         X = data[:self.x_frames]
#         y = data[self.x_frames:]
#
#         return X, y


class StockDataset(Dataset):

    def __init__(self, symbol, x_frames, y_frames, start, end):
        self.symbol = symbol
        self.x_frames = x_frames
        self.y_frames = y_frames

        self.start = datetime.datetime(*start)
        self.end = datetime.datetime(*end)

        self.data = pdr.DataReader(self.symbol, 'yahoo', self.start, self.end)
        print(self.data.isna().sum())
    ## 데이터셋에 len() 을 사용하기 위해 만들어주는것 (dataloader에서 batch를 만들때 이용됨)
    def __len__(self):
        return len(self.data) - (self.x_frames + self.y_frames) + 1

    ## a[:]와 같은 indexing 을 위해 getinem 을 만듬
    ## custom dataset이 list가 아님에도 그 데이터셋의 i번째의 x,y를 출력해줌
    def __getitem__(self, idx):
        idx += self.x_frames
        ## iloc
        data = self.data.iloc[idx - self.x_frames:idx + self.y_frames]
        #data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] ## 컬럼순서맞추기 위해 한것
        data = data[['Close']]
        ## nomalization
        data = data.apply(lambda x: np.log(x + 1) - np.log(x[self.x_frames - 1] + 1))
        data = data.values ## (data.frame >> numpy array) convert >> 나중에 dataloader가 취합해줌
        ## x와 y 기준으로 split
        X = data[:self.x_frames]
        y = data[self.x_frames:]

        return X, y




class csvStockDataset(Dataset):

    def __init__(self, data_site , x_frames, y_frames):
        self.data_site = data_site
        self.x_frames = x_frames
        self.y_frames = y_frames

        self.data = self.csvdata(self.data_site)

        print(self.data.isna().sum())
    def csvdata(self, data_site):
        f = open(data_site, 'r', encoding='cp949')
        rdr = csv.reader(f)
        data = []
        for line in rdr:
            # print(line)
            data.append(line)
        f.close()
        data = pd.DataFrame(data)
        print(data)
        data_columns = list(data.iloc[0,:])
        data = data.iloc[1:,:]
        data_date = list(data.iloc[:,0])
        data_time_ind = pd.DatetimeIndex(data_date)
        data = data.iloc[:, 1:7]
        df = pd.DataFrame(np.array(data), index=data_time_ind, columns=data_columns[1:])
        return df

    ## 데이터셋에 len() 을 사용하기 위해 만들어주는것 (dataloader에서 batch를 만들때 이용됨)
    def __len__(self):
        return len(self.data) - (self.x_frames + self.y_frames) + 1

    ## a[:]와 같은 indexing 을 위해 getinem 을 만듬
    ## custom dataset이 list가 아님에도 그 데이터셋의 i번째의 x,y를 출력해줌
    def __getitem__(self, idx):
        idx += self.x_frames
        ## iloc
        print('data :{}, shape : {}'.format(self.data, self.data.shape))
        data = self.data.iloc[idx - self.x_frames:idx + self.y_frames]
        #data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] ## 컬럼순서맞추기 위해 한것
        print('data :{}, shape : {}'.format(data, data.shape))
        data = data.loc[:,'종가']
        print('data :{}, shape : {}'.format(data, data.shape))
        ## nomalization
        data = data.apply(lambda x: np.log(x + 1) - np.log(x[self.x_frames - 1] + 1))
        print('data :{}, shape : {}'.format(data, data.shape))
        data = data.values ## (data.frame >> numpy array) convert >> 나중에 dataloader가 취합해줌
        ## x와 y 기준으로 split
        X = data[:self.x_frames]
        y = data[self.x_frames:]

        return X, y

class csvStockDataset(Dataset):

    def __init__(self, data_site , x_frames, y_frames, start, end):
        self.data_site = data_site
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.start = start
        self.end = end

        self.data = self.csvdata(self.data_site)

        print(self.data.isna().sum())
    def csvdata(self, data_site):
        f = open(data_site, 'r', encoding='cp949')
        rdr = csv.reader(f)
        data = []
        for line in rdr:
            # print(line)
            data.append(line)
        f.close()
        data = pd.DataFrame(data)
        print(data)
        data_columns = list(data.iloc[0,:])
        data = data.iloc[1:,:]
        data_date = list(data.iloc[:,0])
        data_time_ind = pd.DatetimeIndex(data_date)
        data = data.iloc[:, 1:7]
        df = pd.DataFrame(np.array(data), index=data_time_ind, columns=data_columns[1:])
        return df

    ## 데이터셋에 len() 을 사용하기 위해 만들어주는것 (dataloader에서 batch를 만들때 이용됨)
    def __len__(self):
        data = self.data[self.start:self.end]
        return len(data) - (self.x_frames + self.y_frames) + 1

    ## a[:]와 같은 indexing 을 위해 getinem 을 만듬
    ## custom dataset이 list가 아님에도 그 데이터셋의 i번째의 x,y를 출력해줌
    def __getitem__(self, idx):
        idx += self.x_frames
        ## iloc
        data = self.data[self.start:self.end]
        data = self.data.iloc[idx - self.x_frames:idx + self.y_frames]
        #data = data[['High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume']] ## 컬럼순서맞추기 위해 한것
        ## nomalization
        data = data.astype({'종가': 'float'})
        data = data.loc[:, '종가']

        data = pd.DataFrame(data)
        data = data.apply(lambda x: np.log(x + 1) - np.log(x[self.x_frames - 1] + 1))
        data = data.values ## (data.frame >> numpy array) convert >> 나중에 dataloader가 취합해줌
        ## x와 y 기준으로 split
        X = data[:self.x_frames]
        y = data[self.x_frames:]

        return X, y


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn

        ## 파이토치에 있는 lstm모듈
        ## output dim 은 self.regressor에서 사용됨
        self.RNN = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)
        self.hidden = self.init_hidden()
        self.regressor = self.make_regressor()

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, requires_grad=True)

    def make_regressor(self):  # 간단한 MLP를 만드는 함수
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))  ##  nn.BatchNorm1d
        layers.append(nn.Dropout(self.dropout))  ##  nn.Dropout
        # print('dropout layers: {}'.format(layers))

        ## hidden dim을 outputdim으로 바꿔주는 MLP
        # layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        # #print('Linear layers: {}'.format(layers))
        # layers.append(nn.ReLU())
        # #print('ReLU layers: {}'.format(layers))
        # layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim)) # 여기서 output dim이 사용됨
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        # print('last Linear layers: {}'.format(layers))
        regressor = nn.Sequential(*layers)
        # print('regressor layers: {}'.format(regressor))
        return regressor

    def forward(self, x):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음

        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state
        RNN_out, self.hidden = self.RNN(x)

        ## lstm_out : 각 time step에서의 lstm 모델의 output 값
        ## lstm_out[-1] : 맨마지막의 아웃풋 값으로 그 다음을 예측하고싶은 것이기 때문에 -1을 해줌
        y_pred = self.regressor(RNN_out[-1].view(self.batch_size, -1))  ## self.batch size로 reshape해 regressor에 대입

        return y_pred


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn

        ## 파이토치에 있는 lstm모듈
        ## output dim 은 self.regressor에서 사용됨
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.hidden = self.init_hidden()
        self.regressor = self.make_regressor()
        #self.fc = nn.Linear(self.input_dim, self.output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def make_regressor(self): # 간단한 MLP를 만드는 함수
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))  ##  nn.BatchNorm1d
        layers.append(nn.Dropout(self.dropout))    ##  nn.Dropout

        ## hidden dim을 outputdim으로 바꿔주는 MLP
        # layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim)) # 여기서 output dim이 사용됨
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor

    def forward(self, x):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음
        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        ## lstm_out : 각 time step에서의 lstm 모델의 output 값
        ## lstm_out[-1] : 맨마지막의 아웃풋 값으로 그 다음을 예측하고싶은 것이기 때문에 -1을 해줌
        y_pred = self.regressor(lstm_out[-1].view(self.batch_size, -1)) ## self.batch size로 reshape해 regressor에 대입
        #y_pred = self.fc(lstm_out[-1].view(self.batch_size, -1))
        return y_pred


class GRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn

        ## 파이토치에 있는 lstm모듈
        ## output dim 은 self.regressor에서 사용됨
        self.GRU = nn.GRU(input_size=self.input_dim, hidden_size= self.hidden_dim, num_layers = self.num_layers)
        self.hidden = self.init_hidden()
        self.regressor = self.make_regressor()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, requires_grad=True))

    def make_regressor(self): # 간단한 MLP를 만드는 함수
        layers = []
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.hidden_dim))  ##  nn.BatchNorm1d
        layers.append(nn.Dropout(self.dropout))    ##  nn.Dropout
        #print('dropout layers: {}'.format(layers))

        ## hidden dim을 outputdim으로 바꿔주는 MLP
        # layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        # #print('Linear layers: {}'.format(layers))
        # layers.append(nn.ReLU())
        # #print('ReLU layers: {}'.format(layers))
        # layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim)) # 여기서 output dim이 사용됨
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        #print('last Linear layers: {}'.format(layers))
        regressor = nn.Sequential(*layers)
        #print('regressor layers: {}'.format(regressor))
        return regressor

    def forward(self, x):
        # 새로 opdate 된 self.hidden과 lstm_out을 return 해줌
        # self.hidden 각각의 layer의 모든 hidden state 를 갖고있음

        ## LSTM의 hidden state에는 tuple로 cell state포함, 0번째는 hidden state tensor, 1번째는 cell state

        GRU_out, self.hidden = self.GRU(x)
        # print('RNN_out : {}, RNN_out shape : {}'.format(RNN_out, RNN_out.shape))
        # print('self.hidden : {}, self.hidden shape : {}'.format(self.hidden, self.hidden.shape))
        ## lstm_out : 각 time step에서의 lstm 모델의 output 값
        ## lstm_out[-1] : 맨마지막의 아웃풋 값으로 그 다음을 예측하고싶은 것이기 때문에 -1을 해줌
        y_pred = self.regressor(GRU_out[-1].view(self.batch_size, -1)) ## self.batch size로 reshape해 regressor에 대입
        # print('rnn_out[-1]: {},rnn_out[-1] shape : {}'.format(RNN_out[-1], RNN_out[-1].shape))
        # print('rnn_out[-1].view(self.batch_size, -1) : {}, rnn_out[-1].view(self.batch_size, -1) shape : {}'.format(RNN_out[-1].view(self.batch_size, -1), RNN_out[-1].view(self.batch_size, -1).shape))
        # print('rnn y_pred: {}, rnn y_pred shape : {}'.format(y_pred, y_pred.shape))

        return y_pred