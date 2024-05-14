import os
import math
import json
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import talib as ta
import qlib
from qlib.data import D
from qlib.constant import REG_US

provider_uri = "F:/qlib/qlib_data/us_data" 
qlib.init(provider_uri=provider_uri, region=REG_US)

def ZscoreNorm(series):
    return (series-np.mean(series))/np.std(series)

def get_base_company(args):
    instruments = D.instruments(market='nasdaq100')
    company_pr = D.list_instruments(instruments=instruments, start_time=args.start_time, end_time=args.end_time, as_list=True)
    company_pr.sort()
    return company_pr

def get_data(start_time, end_time, selected_tickers, market='sp500'):
    if selected_tickers is None:
        instruments = D.instruments(market=market)
        all_tickers = (D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True))
        # get selected tickers and selected data
        selected_tickers = all_tickers 
    else:
        instruments = selected_tickers

    # get timestamps
    all_timestamps = list(D.calendar(start_time=start_time, end_time=end_time))
    
    # get data
    all_data = pd.read_csv('./dataset/all_data.csv')
    all_data = all_data.dropna()

    for timestamp in all_timestamps:
        tickers = list(all_data.loc[timestamp].index)
        union = list(set(tickers) & set(selected_tickers))
        selected_tickers = union

    selected_tickers.sort(reverse=False)
    selected_data = all_data.loc[(slice(None), selected_tickers), :]

    # examine alignment
    if len(selected_data) == len(all_timestamps) * len(selected_tickers):
        return all_timestamps, selected_tickers, selected_data
    else:
        raise Exception('Data is not aligned.')

def add_alpha(args, selected_tickers):
    print('Loading base technical data...')
    all_timestamps, all_tickers, all_data = get_data(
        start_time=args.prestart_time, 
        end_time=args.lagend_time, 
        selected_tickers=selected_tickers, 
        market='sp500',
        )
    data_with_alpha = all_data.copy()
    
    print('Loading indicators...')
    for comp in all_tickers:
        close_series = all_data.loc[:,'feature'].loc[:,'$close'].swaplevel().loc[comp, :]
        high_series = all_data.loc[:,'feature'].loc[:,'$high'].swaplevel().loc[comp, :]
        low_series = all_data.loc[:,'feature'].loc[:,'$low'].swaplevel().loc[comp, :]
        volume_series = all_data.loc[:,'feature'].loc[:,'$volume'].swaplevel().loc[comp, :]
        return_series = all_data.loc[:,'feature'].loc[:,'$close/Ref($close, 1)-1'].swaplevel().loc[comp, :]

        df_alpha = pd.DataFrame(close_series)
        types_all = ['SMA','EMA','WMA','DEMA','TEMA','TRIMA','KAMA','MAMA','T3',
                    'atr', 'natr', 'trange',
                    'rsi',

                    'obv', 'norm_return', 'obv-ref',
                    'macd', 'macdsignal', 'macdhist', 'macdhist-ref',
                    'slowk', 'slowd', 'norm_kdjhist', 'kdjhist-ref',]
        
        # MA
        types_ma=['SMA','EMA','WMA','DEMA','TEMA','TRIMA','KAMA','MAMA','T3']
        for i in range(len(types_ma)):
            df_alpha[types_ma[i]] = ZscoreNorm(ta.MA(close_series, timeperiod=5, matype=i))
        # Volatility Indicators
        df_alpha['atr'] = ZscoreNorm(ta.ATR(high_series, low_series, close_series, timeperiod=5))
        df_alpha['natr'] = ZscoreNorm(ta.NATR(high_series, low_series, close_series, timeperiod=5))
        df_alpha['trange'] = ZscoreNorm(ta.TRANGE(high_series, low_series, close_series))
        # RSI
        df_alpha['rsi'] = ta.RSI(close_series, timeperiod=5) / 100

        # Volume Indicators
        df_alpha['obv'] = ZscoreNorm(ta.OBV(close_series, volume_series))
        df_alpha['norm_return'] = ZscoreNorm(return_series)
        df_alpha['obv-ref'] = df_alpha['obv'] - df_alpha['obv'].shift(1)
        # MACD
        macd, macdsignal, macdhist = ta.MACD(close_series, fastperiod=12, slowperiod=26, signalperiod=9)
        df_alpha['macd'], df_alpha['macdsignal'], df_alpha['macdhist'] = ZscoreNorm(macd), ZscoreNorm(macdsignal), ZscoreNorm(macdhist)
        df_alpha['macdhist-ref'] = ZscoreNorm(df_alpha['macdhist'] - df_alpha['macdhist'].shift(1))
        # KDJ
        slowk, slowd = ta.STOCH(high_series, low_series, close_series, fastk_period=5, slowk_period=3)
        df_alpha['slowk'] = ZscoreNorm(slowk)
        df_alpha['slowd'] = ZscoreNorm(slowd)
        df_alpha['norm_kdjhist'] = ZscoreNorm(slowk - slowd)
        df_alpha['kdjhist-ref'] = ZscoreNorm((slowk-slowd) - (slowk-slowd).shift(1))


        newindex = pd.MultiIndex.from_product([df_alpha.index.to_list(), [comp]], names=['datetime', 'instrument'])
        df_alpha.set_index(newindex, inplace=True)
        for type in types_all:
            data_with_alpha.loc[(slice(None), comp), ('alpha', type)] = df_alpha[type]
            
    data_with_alpha = data_with_alpha.loc[datetime.datetime.strptime(args.start_time, '%Y-%m-%d'):datetime.datetime.strptime(args.end_time, '%Y-%m-%d')]
    if pd.isnull(data_with_alpha.values).any():
        print('Exist Nan')

    final_timestamps = list(D.calendar(start_time=args.start_time, end_time=args.end_time))
    if len(data_with_alpha) == len(final_timestamps) * len(all_tickers):
        return final_timestamps, all_tickers, data_with_alpha
    else:
        raise Exception('Data is not aligned.')
    
def get_features_n_labels(args, selected_tickers):
    final_timestamps, all_tickers, data_with_alpha = add_alpha(args, selected_tickers=selected_tickers)

    num_times = len(final_timestamps)
    num_nodes = len(all_tickers)
    num_features_n_label = data_with_alpha.shape[1]

    raw_data = torch.Tensor(data_with_alpha.values)
    features_n_labels = raw_data.reshape(num_times, num_nodes, num_features_n_label)
    features = features_n_labels[:, :, 1:]
    labels = features_n_labels[:, :, 0]
    return features, labels, all_tickers, final_timestamps

def get_windows_mpa(inputs, targets, dataset, mode, device, args, shuffle=False):
    window_size = args.window_size
    lambda_param = 11.62

    ## poisson sampling
    unmask_num = 0
    while(unmask_num==0):
        unmask_num = np.random.poisson(lambda_param)
    
    pred_acc = args.pred_acc
    
    time_length = len(inputs) - window_size + 1
    if dataset == 'train':
        start = 0
        end = math.floor(time_length / 10 * 8)
    elif dataset == 'valid':
        start = math.floor(time_length / 10 * 8) + 1
        end = math.floor(time_length / 10 * 9)
    elif dataset == 'test':
        start = math.floor(time_length / 10 * 9) + 1
        end = time_length - 1
    else:
        raise Exception('Unknown dataset.')
    length = end - start + 1
    if shuffle == True:
        indexs = torch.randperm(length)
    elif shuffle == False:
        indexs = range(length)

    for index in indexs:
        i = index + start
        window = inputs[i:(i+window_size), :, :]
        x_window = window.permute(1, 0, 2).to(device).squeeze()
        y_window = targets[i+window_size-1].to(device).squeeze()

        ## noise addition
        confidence = torch.Tensor(np.random.uniform(0, 0.5, size=y_window.size(0)))
        y_window_noised = torch.abs(y_window - confidence)

        if mode == 'pre-train':
            mask = []
            pop_ids = []
            for i in range(y_window.shape[0]):
                mask.append(i)

            for i in range(unmask_num):
                pop_id = torch.randint(low=0, high=len(mask), size=[1])
                pop_ids.append(mask[pop_id])
                mask.pop(pop_id)

            mask = torch.Tensor(mask).long()
            unmask = torch.Tensor(pop_ids).long()

            hashtag_mask = torch.zeros_like(y_window)
            hashtag_mask[mask] = 1
            hashtag_up = y_window_noised.clone().detach()
            hashtag_up[mask] = 0
            hashtag_down = (1 - y_window_noised).clone().detach()
            hashtag_down[mask] = 0

            ## mutation
            if dataset != 'train':
                for id in unmask:
                    if torch.rand(1) > pred_acc:
                        hashtag_up[id] = (1 - hashtag_up[id]).clone().detach()
                        hashtag_down[id] = (1 - hashtag_down[id]).clone().detach()

            x_tag = torch.cat([hashtag_down.unsqueeze(1), hashtag_up.unsqueeze(1), hashtag_mask.unsqueeze(1)], dim=-1)
            yield x_window, x_tag, y_window, mask

def get_windows_hmon(inputs, message_features, targets, dataset, device, shuffle=False):
    window_size = 12
    
    time_length = len(inputs) - window_size + 1
    if dataset == 'train':
        start = 0
        end = math.floor(time_length / 10 * 8)
    elif dataset == 'valid':
        start = math.floor(time_length / 10 * 8) + 1
        end = math.floor(time_length / 10 * 9)
    elif dataset == 'test':
        start = math.floor(time_length / 10 * 9) + 1
        end = time_length - 1
    else:
        raise Exception('Unknown dataset.')
    length = end - start + 1
    if shuffle == True:
        indexs = torch.randperm(length)
    elif shuffle == False:
        indexs = range(length)

    for index in indexs:
        i = index + start
        window = inputs[i:(i+window_size), :, :]
        x_window = window.permute(1, 0, 2).to(device).squeeze()
        x_message = message_features[i+window_size-1].to(device).squeeze()
        y_window = targets[i+window_size-1].to(device).squeeze()

        yield x_window, x_message, y_window

def get_idx(stocks, company_final):
    idxes = []
    for stock in stocks:
        idx = company_final.index(stock)
        idxes.append(idx)
    return idxes

def get_news(news_path, dim_emb, final_timestamps, company_final):
    all_news = []
    for date_dt in tqdm(final_timestamps):
        stocks_news = torch.zeros([len(company_final), dim_emb])
        related_stocks = []
        date = date_dt.strftime('%Y-%m-%d')
        file_name = date + '.json'
        file = os.path.join(news_path, file_name)

        if not os.path.exists(file):
            all_news.append(stocks_news)
            continue
        else:
            with open(file, 'r') as f:
                news = json.load(f)
        for new in news:
            for stocks in new['stocks']:
                stock = stocks['name']
                if stock in company_final:
                    related_stocks.append(stock)
        related_stocks = list(set(related_stocks))
        related_stocks.sort()
        idxes = get_idx(related_stocks, company_final)
        for new in news:
            new_stocks = []
            for stocks in new['stocks']:
                new_stocks.append(stocks['name'])
        for i, tar_stock in enumerate(related_stocks):
            stock_news = []
            for new in news:
                new_stocks = []
                for stocks in new['stocks']:
                    new_stocks.append(stocks['name'])           
                if tar_stock in new_stocks:
                    new_emb = torch.Tensor(new['pooler_output']).squeeze()
                    stock_news.append(new_emb)
            multinews = torch.stack(stock_news, dim=0)
            count = multinews.size(0)
            stock_multinews = multinews.sum(dim=0)/count
            stocks_news[idxes[i], :] = stock_multinews
        all_news.append(stocks_news)
    message_features = torch.stack(all_news, dim=0)
    return message_features