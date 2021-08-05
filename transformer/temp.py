import math
import sys
sys.path.append('C:\\Users\\leete\\PycharmProjects\\transformer')
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import argparse
from copy import deepcopy # Add Deepcopy for args
import matplotlib.pyplot as plt
import LSTM_MODEL_DATASET as LSTMMD
from LSTM_MODEL_DATASET import metric1 as metric1
from LSTM_MODEL_DATASET import metric2 as metric2
from LSTM_MODEL_DATASET import metric3 as metric3
from LSTM_MODEL_DATASET import StockDatasetCV
from LSTM_MODEL_DATASET import CV_Data_Spliter
from LSTM_MODEL_DATASET import CV_train_Spliter
import os
import csv


def train(transformer, partition, transformer_optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],  ## DataLoader는 dataset에서 불러온 값으로 랜덤으로 배치를 만들어줌
                             batch_size=args.batch_size,
                             shuffle=False, drop_last=True)

    transformer.train()
    transformer.zero_grad()
    transformer.zero_grad()

    not_used_data_len = len(partition['train']) % args.batch_size
    train_loss = 0.0
    y_pred_graph = []

    for i, (X, y, min, max) in enumerate(trainloader):
        ## (batch size, sequence length, input dim)
        ## x = (10, n, 6) >> x는 n일간의 input
        ## y= (10, m, 1) or (10, m)  >> y는 m일간의 종가를 동시에 예측
        ## lstm은 한 스텝별로 forward로 진행을 함
        ## (sequence length, batch size, input dim) >> 파이토치 default lstm은 첫번째 인자를 sequence length로 받음
        ## x : [n, 10, 6], y : [m, 10]
        X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)

        ## transpose는 seq length가 먼저 나와야 하기 때문에 0번째와 1번째를 swaping
        transformer.zero_grad()
        transformer_optimizer.zero_grad()
        #transformer.hidden = [hidden.to(args.device) for hidden in transformer.init_hidden()]
        y_pred = transformer(X,y)
        #decoder_hidden = encoder_hidden
        y_true = y[:, :].float().to(args.device)  ## index-3은 종가를 의미(dataframe 상에서)

        # print(torch.max(X[:, :, 3]), torch.max(y_true))

        max = max.to(args.device)
        min = min.to(args.device)

        reformed_y_pred = y_pred.squeeze() * (max - min) + min

        y_pred_graph = y_pred_graph + reformed_y_pred.tolist()

        loss = loss_fn(y_pred.view(-1), y_true.view(-1))  # .view(-1)은 1열로 줄세운것

        loss.backward()  ## gradient 계산

        transformer_optimizer.step()  ## parameter 갱신 parameter를 update 해줌 (.backward() 연산이 시행된다면(기울기 계산단계가 지나가면))

        train_loss += loss.item()  ## item()은 loss의 스칼라값을 칭하기때문에 cpu로 다시 넘겨줄 필요가 없다.

    train_loss = train_loss / len(trainloader)
    return transformer, train_loss, y_pred_graph, not_used_data_len


def validate(transformer, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)

    not_used_data_len = len(partition['val']) % args.batch_size

    transformer.eval()

    val_loss = 0.0
    with torch.no_grad():
        y_pred_graph = []
        for i, (X, y, min, max) in enumerate(valloader):
            X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)
            #encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]

            y_pred = transformer(X,y)

            y_true = y[:, :].float().to(args.device)

            max = max.to(args.device)
            min = min.to(args.device)

            reformed_y_pred = y_pred.squeeze() * (max - min) + min
            y_pred_graph = y_pred_graph + reformed_y_pred.tolist()

            # print('validate y_pred: {}, y_pred.shape : {}'. format(y_pred, y_pred.shape))
            loss = loss_fn(y_pred.view(-1), y_true.view(-1))

            val_loss += loss.item()

    val_loss = val_loss / len(valloader)  ## 한 배치마다의 로스의 평균을 냄
    return val_loss, y_pred_graph, not_used_data_len  ## 그결과값이 한 에폭마다의 LOSS

def test(transformer, partition, args):
    testloader = DataLoader(partition['test'],
                            batch_size=args.batch_size,
                            shuffle=False, drop_last=True)
    not_used_data_len = len(partition['test']) % args.batch_size

    transformer.eval()

    test_loss_metric1 = 0.0
    test_loss_metric2 = 0.0
    test_loss_metric3 = 0.0
    with torch.no_grad():
        y_pred_graph = []
        for i, (X, y, min, max) in enumerate(testloader):
            X = X.transpose(0, 1).unsqueeze(-1).float().to(args.device)
            # encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]

            y_pred= transformer(X,y)

            y_true = y[:, :].float().to(args.device)

            max = max.to(args.device)
            min = min.to(args.device)

            reformed_y_pred = y_pred.squeeze() * (max - min) + min
            reformed_y_true = y_true.squeeze() * (max - min) + min
            y_pred_graph = y_pred_graph + reformed_y_pred.tolist()

            test_loss_metric1 += metric1(reformed_y_pred, reformed_y_true)
            test_loss_metric2 += metric2(reformed_y_pred, reformed_y_true)
            test_loss_metric3 += metric3(reformed_y_pred, reformed_y_true)

    test_loss_metric1 = test_loss_metric1 / len(testloader)
    test_loss_metric2 = test_loss_metric2 / len(testloader)
    test_loss_metric3 = test_loss_metric3 / len(testloader)
    return test_loss_metric1, test_loss_metric2, test_loss_metric3, y_pred_graph, not_used_data_len


def experiment(partition, args):
    transformer = args.transformer(args.feature_size, args.dropout)
    transformer.to(args.device)

    loss_fn = nn.MSELoss()
    # loss_fn.to(args.device) ## gpu로 보내줌  간혹 loss에 따라 안되는 경우도 있음
    if args.optim == 'SGD':
        transformer_optimizer = optim.SGD(transformer.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        transformer_optimizer = optim.RMSprop(transformer.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        transformer_optimizer = optim.Adam(transformer.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    # ===================================== #
    ## 우리는 지금 epoch 마다 모델을 저장해야 하기때문에 여기에 저장하는 기능을 넣어야함.
    ## 실제로 우리는 디렉토리를 만들어야함
    ## 모델마다의 디렉토리를 만들어야하는데
    epoch_graph_list = []

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()

        # def train(transformer, partition, transformer_optimizer, loss_fn, args):
        transformer, train_loss, graph1, unused_triain = train(transformer, partition, transformer_optimizer, loss_fn, args)
        # def validate(transformer, partition, loss_fn, args):
        val_loss, graph2, unused_val = validate(transformer, partition, loss_fn, args)

        te = time.time()

        epoch_graph_list.append([graph1, graph2])
        # ====== Add Epoch Data ====== # ## 나중에 그림그리는데 사용할것
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # ============================ #
        ## 각 에폭마다 모델을 저장하기 위한 코드
        torch.save(transformer.state_dict(), args.innate_path + '\\' + str(epoch) + '_epoch' + '_transformer' + '.pt')

        print('Epoch {}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec, Iteration {}'
              .format(epoch, train_loss, val_loss, te - ts, args.iteration))

    ## 여기서 구하는것은 val_losses에서 가장 값이 최소인 위치를 저장함
    site_val_losses = val_losses.index(min(val_losses))  ## 10 epoch일 경우 0번째~9번째 까지로 나옴
    transformer = args.transformer(args.feature_size, args.dropout)

    transformer.to(args.device)

    transformer.load_state_dict(torch.load(args.innate_path + '\\' + str(site_val_losses) + '_epoch' + '_transformer' + '.pt'))

    ## graph
    train_val_graph = epoch_graph_list[site_val_losses]
    # def test(transformer, partition, args):
    test_loss_metric1, test_loss_metric2, test_loss_metric3, graph3, unused_test = test(transformer, partition, args)
    print('test_loss_metric1: {},\n test_loss_metric2: {}, \ntest_loss_metric3: {}'
          .format(test_loss_metric1, test_loss_metric2, test_loss_metric3))

    with open(args.innate_path + '\\' + str(site_val_losses) + 'Epoch_test_metric' + '.csv', 'w') as fd:
        print('test_loss_metric1 : {} \n test_loss_metric2 : {} \n test_loss_metric3 : {}'
              .format(test_loss_metric1, test_loss_metric2, test_loss_metric3), file=fd)
    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['test_loss_metric1'] = test_loss_metric1
    result['test_loss_metric2'] = test_loss_metric2
    result['test_loss_metric3'] = test_loss_metric3
    result['train_val_graph'] = train_val_graph
    result['test_graph'] = graph3
    result['unused_data'] = [unused_triain, unused_val, unused_test]

    return vars(args), result  ## vars(args) 1: args에있는 attrubute들을 dictionary 형태로 보길 원한다면 vars 함

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)   # torch.Size([max_len, 1, d_model])
        #pe.requires_grad = False
        self.register_buffer('pe', pe) ## 매개변수로 간주하지 않기 위한 것

    def forward(self, x):
        return x + self.pe[:x.size(0), :]



class TransAm(nn.Module):
    def __init__(self, feature_size, dropout):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.tgt_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
        # num_decoder_layers: int = 6,
        # d_model > 임베딩 차원
        ## dim_feedforward: int = 2048 >> feed forward linear layer 의 차원
        self.transformer = nn.Transformer(d_model=feature_size, nhead=4, dropout=dropout,
                                          num_encoder_layers=1,num_decoder_layers=1)
        self.decoder = nn.Linear(feature_size, 1)

        self.linear = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):

        tgt = tgt.unsqueeze(0).to(torch.float32)

        # torch.Size([20, 128, 1])
        # torch.Size([128, 1])
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
            self.tgt_mask = mask

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # torch.Size([20, 128, 16])
        # torch.Size([1, 128, 16])
        # output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        # output = self.decoder(output)
        output = self.transformer(src, tgt, src_mask=self.src_mask, tgt_mask = self.tgt_mask)

        return self.linear(output)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)                ## 하삼각행렬
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")  ## ""을 써주는 이유는 터미널이 아니기때문에
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
args.batch_size = 128
args.x_frames = 20
args.y_frames = 1
args.transformer = TransAm

# ====== Model Capacity ===== #
args.input_dim = 1
args.feature_size = 16
args.n_layers = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.0
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam'  # 'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 2
args.split = 2
# ====== Experiment Variable ====== #


# '^KS11' : KOSPI                                      'KS11'
# '^KQ11' : 코스닥                                      'KQ11'
# '^IXIC' : 나스닥                                      'IXIC'
# '^GSPC' : SNP 500 지수                                'US500'
# '^DJI' : 다우존수 산업지수                              'DJI'
# '^HSI' : 홍콩 항생 지수                                'HK50'
# '^N225' : 니케이지수                                   'JP225'
# '^GDAXI' : 독일 DAX                                   'DE30'
# '^FTSE' : 영국 FTSE                                   'UK100'
# '^FCHI' : 프랑스 CAC                                  'FCHI'
# '^IBEX' : 스페인 IBEX
# '^TWII' : 대만 기권                                   'TWII'
# '^AEX' : 네덜란드 AEX
# '^BSESN' : 인도 센섹스
# 'RTSI.ME' : 러시아 RTXI
# '^BVSP' : 브라질 보베스파 지수
# 'GC=F' : 금 가격                                       'GC'
# 'CL=F' : 원유 가격 (2000/ 8 / 20일 부터 데이터가 있음)    'CL'
# 'BTC-USD' : 비트코인 암호화폐                           'BTC/KRW'
# 'ETH-USD' : 이더리움 암호화폐                           'ETH/KRW'
## 중국                                                 'CSI300'
# 	상해 종합                                            'SSEC'
#  베트남 하노이                                          'HNX30'


# model_list = [LSTMMD.RNN,LSTMMD.LSTM,LSTMMD.GRU]
# data_list = ['^KS11', '^KQ11','^IXIC','^GSPC','^DJI','^HSI',
#              '^N225','^GDAXI','^FCHI','^IBEX','^TWII','^AEX',
#              '^BSESN','^BVSP','GC=F','BTC-USD','ETH-USD','CL=F']

data_list = ['KS11', 'KQ11', 'IXIC', 'US500',
             'DJI', 'HK50', 'JP225', 'DE30',
             'UK100', 'FCHI', 'TWII', 'GC',
             'CL', 'CSI300',
             'SSEC', 'HNX30']
data_list = ['KS11']
args.save_file_path = 'C:\\Users\\leete\\PycharmProjects\\transformer\\results'

with open(args.save_file_path + '\\' + 'transformer_result_t.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)

    wr.writerow(["model", "stock", "entire_exp_time",  "avg_test_metric1", "std_test_metric1",
                 "avg_test_metric2", "std_test_metric2",
                 "avg_test_metric3", "std_test_metric3"])

    for j in data_list:
        setattr(args, 'symbol', j)
        model_name = "transformer"
        args.new_file_path = args.save_file_path + '\\' + model_name + '_' + args.symbol
        os.makedirs(args.new_file_path)
        if args.symbol == 'KS11':
            data_start = '2013-03-03'  # (2013, 3, 3)
            data_end = '2020-12-31'  # (2020, 12, 31)
        elif args.symbol == 'CL':
            data_start = '2011-01-01'  # (2011, 1, 1)               ##(2000, 8, 23)
            data_end = '2020-12-31'  # (2020, 12, 31)
        elif args.symbol == 'BTC-USD':
            data_start = '2014-09-17'  # (2014, 9, 17)
            data_end = '2020-12-31'  # (2020, 12, 31)
        elif args.symbol == 'ETH-USD':
            data_start = '2015-08-07'  # (2015, 8, 7)
            data_end = '2020-12-31'  # (2020, 12, 31)
        else:  ## 나머지 모든 데이터들
            data_start = '2011-01-01'  # (2011, 1, 1)
            data_end = '2020-12-31'  # (2020, 12, 31)

        est = time.time()

        splitted_test_train = CV_Data_Spliter(args.symbol, data_start, data_end, n_splits=args.split)
        entire_data = splitted_test_train.entire_data()

        args.series_Data = splitted_test_train.entire_data
        test_metric1_list = []
        test_metric2_list = []
        test_metric3_list = []
        for iteration_n in range(args.split):
            args.iteration = iteration_n
            train_data, test_data = splitted_test_train[args.iteration][0], splitted_test_train[args.iteration][1]
            test_size = splitted_test_train.test_size
            splitted_train_val = CV_train_Spliter(train_data, args.symbol, test_size=test_size)
            train_data, val_data = splitted_train_val[1][0], splitted_train_val[1][1]

            trainset = StockDatasetCV(train_data, args.x_frames, args.y_frames)
            valset = StockDatasetCV(val_data, args.x_frames, args.y_frames)
            testset = StockDatasetCV(test_data, args.x_frames, args.y_frames)
            partition = {'train': trainset, 'val': valset, 'test': testset}

            args.innate_path = args.new_file_path + '\\' + str(args.iteration) + '_iter'  ## 내부 파일경로
            os.makedirs(args.innate_path)
            print(args)

            setting, result = experiment(partition, deepcopy(args))
            test_metric1_list.append(result['test_loss_metric1'])
            test_metric2_list.append(result['test_loss_metric2'])
            test_metric3_list.append(result['test_loss_metric3'])

            ## 그림
            fig = plt.figure()
            plt.plot(result['train_losses'])
            plt.plot(result['val_losses'])
            plt.legend(['train_losses', 'val_losses'], fontsize=15)
            plt.xlabel('epoch', fontsize=15)
            plt.ylabel('loss', fontsize=15)
            plt.grid()
            plt.savefig(args.new_file_path + '\\' + str(args.iteration) + '_fig' + '.png')
            plt.close(fig)

            predicted_traing = result['train_val_graph'][0]
            predicted_valg = result['train_val_graph'][1]
            predicted_testg = result['test_graph']
            entire_dataa = entire_data['Close'].values.tolist()

            train_length = len(predicted_traing)
            val_length = len(predicted_valg)
            test_length = len(predicted_testg)
            entire_length = len(entire_dataa)

            unused_triain = result['unused_data'][0]
            unused_val = result['unused_data'][1]
            unused_test = result['unused_data'][2]

            train_index = list(range(args.x_frames, args.x_frames + train_length))
            val_index = list(range(args.x_frames + train_length + unused_triain + args.x_frames,
                                   args.x_frames + train_length + unused_triain + args.x_frames + val_length))
            test_index = list(range(
                args.x_frames + train_length + unused_triain + args.x_frames + val_length + unused_val + args.x_frames,
                args.x_frames + train_length + unused_triain + args.x_frames + val_length + unused_val + args.x_frames + test_length))
            entire_index = list(range(entire_length))

            fig2 = plt.figure()
            plt.plot(entire_index, entire_dataa)
            plt.plot(train_index, predicted_traing)
            plt.plot(val_index, predicted_valg)
            plt.plot(test_index, predicted_testg)
            plt.legend(['raw_data', 'predicted_train', 'predicted_val', 'predicted_test'], fontsize=15)
            plt.xlim(0, entire_length)
            plt.xlabel('time', fontsize=15)
            plt.ylabel('value', fontsize=15)
            plt.grid()
            plt.savefig(args.new_file_path + '\\' + str(args.iteration) + '_chart_fig' + '.png')
            plt.close(fig2)

            # save_exp_result(setting, result)

        eet = time.time()

        entire_exp_time = eet - est

        avg_test_metric1 = sum(test_metric1_list) / len(test_metric1_list)
        avg_test_metric2 = sum(test_metric2_list) / len(test_metric2_list)
        avg_test_metric3 = sum(test_metric3_list) / len(test_metric3_list)
        std_test_metric1 = np.std(test_metric1_list)
        std_test_metric2 = np.std(test_metric2_list)
        std_test_metric3 = np.std(test_metric3_list)

        # csv파일에 기록하기
        wr.writerow([model_name, args.symbol,entire_exp_time, avg_test_metric1, std_test_metric1,
                     avg_test_metric2, std_test_metric2,
                     avg_test_metric3, std_test_metric3])

        with open(args.new_file_path + '\\' + 'result_t.txt', 'w') as fd:
            print('metric1 \n avg: {}, std : {}\n'.format(avg_test_metric1, std_test_metric1), file=fd)
            print('metric2 \n avg: {}, std : {}\n'.format(avg_test_metric2, std_test_metric2), file=fd)
            print('metric3 \n avg: {}, std : {}\n'.format(avg_test_metric3, std_test_metric3), file=fd)
        print('{}_{} 30 avg_test_value_list : {}'.format(model_name, args.symbol, avg_test_metric1))
        print('{}_{} 30 avg_test_value_list : {}'.format(model_name, args.symbol, avg_test_metric2))
        print('{}_{} 30 avg_test_value_list : {}'.format(model_name, args.symbol, avg_test_metric3))

