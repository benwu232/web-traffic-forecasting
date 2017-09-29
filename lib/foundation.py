import math
import re
import numpy as np
import pandas as pd
import datetime as dt
import copy
import os, sys
import pickle
from collections import OrderedDict
import csv
import numba
import torch
from torch.autograd import Variable
import logging
import zipfile

agent_type = {
    'spider': 1,
    'user': 2,
    'all-agents': 3
}

access_type = {
    'desktop': 1,
    'mobile-web': 2,
    'all-access': 3
}

categories = {
    'www.mediawiki.org': 1,
    'commons.wikimedia.org': 2,
    'ja.wikipedia.org': 3,
    'fr.wikipedia.org': 4,
    'ru.wikipedia.org': 5,
    'es.wikipedia.org': 6,
    'de.wikipedia.org': 7,
    'zh.wikipedia.org': 8,
    'en.wikipedia.org': 9,
}

def save_dump(dump_data, out_file):
    with open(out_file, 'wb') as fp:
        print('Writing to %s.' % out_file)
        #pickle.dump(dump_data, fp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dump_data, fp)

def load_dump(dump_file):
    with open(dump_file, 'rb') as fp:
        dump = pickle.load(fp)
        return dump

@numba.jit
def cal_smape(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    out = 0
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        c = a+b
        if c == 0:
            continue
        out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out

@numba.jit
def smape_np(y_true, y_pred):
    tmp = (np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true)))
    tmp[np.isnan(tmp)] = 0.0
    return np.mean(tmp)
    #return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))).fillna(0))

@numba.jit
def standardize_mean(y):
    #using deepcopy to keep the original value
    x = copy.deepcopy(y)
    #x = np.nan_to_num(x)
    avg = np.mean(x)
    std = (np.std(x) + 1e-7)
    x -= avg
    x /= std
    x = np.nan_to_num(x)
    return x, avg, std

@numba.jit
def std_median(x):
    median = np.median(x)
    std = x - median
    std = np.square(std)
    len_x = len(std)
    std = np.sum(std)
    std /= len_x
    return std, median

@numba.jit
def standardize_median(y):
    #using deepcopy to keep the original value
    x = copy.deepcopy(y)
    #x = np.nan_to_num(x)
    std_m, median = std_median(x)
    x -= median
    std_m += 1e-8
    x /= std_m
    x = np.nan_to_num(x)
    return x, median, std_m

def standardize_dumb(x):
    return x, 0.0, 1.0

#standardize = standardize_mean
#standardize = standardize_median
standardize = standardize_dumb

df_file = '../input/df.pkl'
def merge_update_data(df, file_name='../input/update_data.pkl'):
    update_data = load_dump(file_name)

    # Add new columns to dataframe
    for key in update_data['1976_Summer_Olympics_en.wikipedia.org_desktop_all-agents'].keys():
        if key not in df.columns:
            df[key] = 0.0

    date_list = list(update_data['1976_Summer_Olympics_en.wikipedia.org_desktop_all-agents'].keys())[1:]
    if update_data['1976_Summer_Olympics_en.wikipedia.org_desktop_all-agents'][date_list[-1]] is None:
        date_list.pop()

    not_update_cnt = 0
    for k in range(len(df)):
        page = df.loc[k]['Page']
        if k % 1000 == 0:
            print('{} Merging: {} ...'.format(k, page))
            print('{} not updated'.format(not_update_cnt))
        if page in update_data:
            for date_str in date_list:
                df.set_value(k, date_str, update_data[page][date_str])
        else:
            not_update_cnt += 1
            #print('{} not in update_data.'.format(page))

    print('Total {} not updated'.format(not_update_cnt))
    #save_dump(df, df_file)
    return df

def load_data(file_name, load_len=-1):
    df = pd.read_csv(file_name)
    #df = df.iloc[:2]
    #df = df.iloc[:70000]
    #df = df.iloc[:10000]
    #df = df.iloc[:300]
    #df = df.iloc[:1000]
    if load_len > 0:
        df = df.iloc[:load_len]
    return df

def pre_process(df):
    df.fillna(method='ffill', inplace=True)
    page_info = df['Page']
    df = df.drop('Page', axis=1)
    return df, page_info

def pre_process1(df):
    df.fillna(method='ffill', inplace=True)
    df_t = df.T
    page_info = df_t.iloc[0]
    df_t = df_t.drop('Page', axis=0)
    df = df_t.T

    return df, page_info

def pre_process1(df, split_num=60):
    df.fillna(method='ffill', inplace=True)
    df_t = df.T
    page_info = df_t.iloc[0]
    df_t = df_t.drop('Page', axis=0)
    if split_num <= 0:
        train = df_t
        test = None
    else:
        train, test = df_t.iloc[:-split_num, :], df_t.iloc[-split_num:, :]

    '''
    m = Prophet()
    train1 = pd.DataFrame()
    train1['ds'] = train.index
    train1['y'] = train.iloc[:, 0].values
    m.fit(train1)

    future = m.make_future_dataframe(periods=N)
    future.tail()

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    '''

    return train, test, page_info

from sklearn.model_selection import KFold
def gen_k_fold(data_np, k_fold=4):
    #kf = KFold(n_splits=k_fold, shuffle=False, random_state=9527)
    kf = KFold(n_splits=k_fold)
    #kf.get_n_splits(data_np)
    row_train_idxes = []
    row_validate_idxes = []
    for train_idx, validate_idx in kf.split(data_np):
        #print("Train:", train_idx, "Validate:", validate_idx)
        row_train_idxes.append(train_idx)
        row_validate_idxes.append(validate_idx)
    return row_train_idxes, row_validate_idxes

def gen_idx_lut(row_idxes, col_start=430, col_end=490):
    idx_lut = []

    for row_idx in row_idxes:
        for col_idx in range(col_start, col_end+1):
            idx_lut.append((row_idx, col_idx))

    return idx_lut




class SmapeLoss1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        seq_len = targets.size()[1]
        batch_size = targets.size()[0]

        smape_sum = 0.0
        for b in range(batch_size):
            smape_value = 0.0
            for k in range(seq_len):
                p = predictions[b, k]
                t = targets[b, k]
                c = torch.abs(p) + torch.abs(t)
                #if torch.abs(c.data[0]) < 1e-3:
                if torch.abs(c).data[0] < 1e-3:
                    continue
                smape_value += torch.abs(t - p) / c
            smape_value *= (200.0 / seq_len)
            #print(smape_value.data[0])
            smape_sum += smape_value
        smape_value = smape_sum / batch_size
        return smape_value


def str2dt(time_str, format='%Y-%m-%d'):
    return dt.datetime.strptime(time_str, format)

#@numba.jit
def gen_key_dict(key_file):
    key_dict_file = '../output/' + os.path.basename(key_file)[:-3] + 'pkl'
    if os.path.isfile(key_dict_file):
        f = open(key_dict_file, 'rb')
        key_dict = pickle.load(f)
        return key_dict

    key_dict = OrderedDict()
    key_df = pd.read_csv(key_file)

    print('Generating key_dict')
    for k in range(len(key_df)):
        if k % 10000 == 0:
            print(k)
        page = key_df.Page.iloc[k][:-11]
        if page not in key_dict:
            key_dict[page] = []
            #print(k)
        key_dict[page].append(key_df.Id.iloc[k])

    f = open(key_dict_file, 'wb')
    pickle.dump(key_dict, f)
    print('Generating key_dict over')

    return key_dict

KEY_DICT = '../output/key_dict.pkl'
KEY_FILE = '../input/key_2.csv'
def gen_submission(page_info, predict_values, submission_file=None, key_file=KEY_FILE):
    key_dict = gen_key_dict(key_file)

    if submission_file is None:
        submission_file = '../output/submission.zip'
        submission_csv = '../output/submission.csv'
    else:
        submission_file = os.path.join('../output', submission_file)
        submission_csv = submission_file[:-3] + 'csv'

    submission_list = []
    for i, page in enumerate(page_info):
        for k, key in enumerate(key_dict[page]):
            submission_list.append([key, round(predict_values[i][k]+0.001, 2)])
    submission_list.sort(key=lambda e: e[-1])

    with open(submission_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Visits'])
        writer.writerows(submission_list)

    #compress
    zout = zipfile.ZipFile(submission_file, "w", zipfile.ZIP_DEFLATED)
    print('Compress {} ...'.format(submission_csv))
    zout.write(submission_csv, arcname=os.path.basename(submission_csv))
    zout.close()
    os.remove(submission_csv)


def cal_weekday(col_idx, offset=3):
    return (col_idx + offset) % 7

def page_info_to_num(page_item):
        results = re.finditer(r'_', page_item)
        positions_ = []
        for r in results:
            positions_.append(r.start())

        agent = page_item[positions_[-1]+1:]
        access = page_item[positions_[-2]+1:positions_[-1]]
        category = page_item[positions_[-3]+1:positions_[-2]]

        agent_num = agent_type[agent]
        access_num = access_type[access]
        category_num = categories[category]

        return agent_num, access_num, category_num

