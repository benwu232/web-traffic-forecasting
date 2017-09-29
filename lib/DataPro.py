import sklearn
import random

from lib.foundation import *


class BatchBuilder2D(object):
    def __init__(self, pro_data, row_idxes, col_idxes, raw_data, batch_size=256, shuffle=True, drop_last=True, target_len=60, target_len_range=[62, 68], target_type=1):
        self.pro_data = pro_data
        self.raw_data = raw_data
        self.col_idxes = col_idxes
        self.col_idxes_len = len(self.col_idxes)
        self.row_idxes_len = len(row_idxes)
        self.row_idxes = np.asarray(row_idxes)
        self.batch_size = batch_size
        self.row_k = 0
        self.col_k = 0
        self.n_features = 1
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.target_len = target_len
        self.target_len_range = list(range(target_len_range[0], target_len_range[1]))
        self.target_type = target_type

    def reset_row_idxes(self):
        self.row_k = 0
        if self.shuffle:
            np.random.shuffle(self.row_idxes)

    def reset_col_idxes(self):
        self.col_k = 0
        if self.shuffle:
            np.random.shuffle(self.col_idxes)

    def gen_batch_idxes(self):
        col_idx = self.col_idxes[self.col_k]
        self.col_k += 1
        if self.col_k >= self.col_idxes_len:
            self.reset_col_idxes()

        new_row_k = self.row_k + self.batch_size
        if new_row_k <= self.row_idxes_len:
            row_idxes = self.row_idxes[self.row_k:new_row_k]
            self.row_k = new_row_k
        elif not self.drop_last:
            row_idxes = self.row_idxes[self.row_k:self.row_idxes_len]
            self.reset_col_idxes()
        else:
            self.reset_col_idxes()
            row_idxes = self.row_idxes[0:self.batch_size]
            self.row_k = self.batch_size
        return row_idxes, col_idx

    #target_type:
    # 0, no target
    # 1, raw target
    # 2, processed target
    def gen_batch(self, pro_data, row_idxes, col_idx, raw_data, target_len=-1):
        #row_idxes, col_idx = self.gen_batch_idxes()
        row_len = pro_data.shape[-1]
        if target_len < 0:
            target_len = random.choice(self.target_len_range)
        target_len = min(target_len, row_len-col_idx)

        batch_size = len(row_idxes)
        seq_len = col_idx
        data_set = np.zeros((seq_len, batch_size, self.n_features))

        avg = np.zeros(batch_size)
        std = np.zeros(batch_size)
        if self.target_type == 0:
            for k, row_idx in enumerate(row_idxes):
                data = pro_data[row_idx, :col_idx]
                #data, avg[k], std[k] = standardize(data)
                data_set[:, k, 0] = np.array(data, dtype=np.float32)
            return [data_set, avg, std]
        elif self.target_type == 2:
            target_set = np.zeros((target_len, batch_size))
            for k, row_idx in enumerate(row_idxes):
                data = pro_data[row_idx, :col_idx]
                data, avg[k], std[k] = standardize(data)
                target = raw_data[row_idx, col_idx:col_idx + target_len]
                target -= avg[k]
                target /= std[k]
                target = np.nan_to_num(target)
                data_set[:, k, 0] = np.array(data, dtype=np.float32)
                target_set[:, k] = np.array(target, dtype=np.float32)
            return [data_set, avg, std, target_set]
        elif self.target_type == 1:
            target_set = np.zeros((target_len, batch_size))
            for k, row_idx in enumerate(row_idxes):
                data = pro_data[row_idx, :col_idx]
                target = raw_data[row_idx, col_idx:col_idx + target_len]
                #target = np.nan_to_num(target)
                data_set[:, k, 0] = np.array(data, dtype=np.float32)
                target_set[:, k] = np.array(target, dtype=np.float32)
            return [data_set, avg, std, target_set]

    def batch4torch(self, np_set):
        if self.target_type == 0:
            return (torch.from_numpy(np_set[0]).float(), np_set[1], np_set[2])
        elif self.target_type == 1:
            return (torch.from_numpy(np_set[0]).float(), np_set[1], np_set[2], torch.from_numpy(np_set[-1]).float())
        elif self.target_type == 2:
            return (torch.from_numpy(np_set[0]), np_set[1], np_set[2], torch.from_numpy(np_set[-1]))

    def build_batch(self):
        row_idxes, col_idx = self.gen_batch_idxes()
        batch_data = self.gen_batch(self.pro_data, row_idxes, col_idx, self.raw_data, target_len=-1)
        return self.batch4torch(batch_data)

    def gen_batch_idxes_once(self):
        row_idxes = None
        col_idx = None
        for col_idx in self.col_idxes:
            row_k = 0
            while row_k < self.row_idxes_len:
                new_row_k = row_k + self.batch_size
                if new_row_k <= self.row_idxes_len:
                    row_idxes = self.row_idxes[row_k:new_row_k]
                    row_k = new_row_k
                elif not self.drop_last:
                    row_idxes = self.row_idxes[row_k:self.row_idxes_len]
                    row_k = self.row_idxes_len
                yield row_idxes, col_idx
        return

    def build_batch_once(self, target_len=-1):
        k = 0
        for row_idxes, col_idx in self.gen_batch_idxes_once():
            #print(k, len(row_idxes), col_idx)
            batch_data = self.gen_batch(self.pro_data, row_idxes, col_idx, self.raw_data, target_len=target_len)
            k += 1
            yield self.batch4torch(batch_data)


class BatchBuilder2DMF(BatchBuilder2D):
    def __init__(self, pro_data, row_idxes, col_idxes, raw_data, batch_size=256, shuffle=True, drop_last=True,
                 target_len=60, target_len_range=[62, 68], target_type=1, page_info=None, n_features=5, pro_mode='train'):
        super().__init__(pro_data, row_idxes, col_idxes, raw_data, batch_size, shuffle, drop_last, target_len, target_len_range, target_type)
        self.page_info = page_info
        self.pro_mode = pro_mode
        self.n_features = n_features

    def cal_weekdays(self, seq_len, week_day_offset=3):
        self.weekdays = np.zeros(seq_len, dtype=np.float32)
        for k in range(seq_len):
            self.weekdays[k] = cal_weekday(k, offset=week_day_offset)

    #target_type:
    # 0, no target
    # 1, raw target
    # 2, processed target
    def gen_batch(self, pro_data, row_idxes, col_idx, raw_data, target_len=-1):
        #row_idxes, col_idx = self.gen_batch_idxes()
        row_len = pro_data.shape[-1]
        if target_len < 0:
            target_len = random.choice(self.target_len_range)
            target_len = min(target_len, row_len-col_idx)

        batch_size = len(row_idxes)
        seq_len = col_idx
        data_set = np.zeros((seq_len, batch_size, self.n_features))

        '''
        if self.pro_mode == 'train':
            self.cal_weekdays(seq_len, 3)
        elif self.pro_mode == 'validate':
            #self.cal_weekdays(row_len - seq_len, seq_len + 3)
            #self.cal_weekdays(target_len, seq_len + 3)
            self.cal_weekdays(seq_len, 3)
        elif self.pro_mode == 'predict':
            self.cal_weekdays(seq_len, 3)
        '''
        self.cal_weekdays(seq_len, 3)

        avg = np.zeros(batch_size)
        std = np.zeros(batch_size)
        if self.target_type == 0:
            for k, row_idx in enumerate(row_idxes):
                agent_num, access_num, cat_num = page_info_to_num(self.page_info[row_idx])
                #data, avg[k], std[k] = standardize(data)
                data_set[:, k, 0] = np.array(pro_data[row_idx, :col_idx], dtype=np.float32)
                data_set[:, k, 1] = np.array(self.weekdays[:col_idx], dtype=np.float32)
                data_set[:, k, 2] = np.ones(seq_len, dtype=np.float32) * agent_num
                data_set[:, k, 3] = np.ones(seq_len, dtype=np.float32) * access_num
                data_set[:, k, 4] = np.ones(seq_len, dtype=np.float32) * cat_num
            return [data_set, avg, std]
        elif self.target_type == 2:
            target_set = np.zeros((target_len, batch_size))
            for k, row_idx in enumerate(row_idxes):
                agent_num, access_num, cat_num = page_info_to_num(self.page_info[row_idx])
                data_set[:, k, 0] = np.array(pro_data[row_idx, :col_idx], dtype=np.float32)
                data_set[:, k, 1] = np.array(self.weekdays[:col_idx], dtype=np.float32)
                data_set[:, k, 2] = np.ones(seq_len, dtype=np.float32) * agent_num
                data_set[:, k, 3] = np.ones(seq_len, dtype=np.float32) * access_num
                data_set[:, k, 4] = np.ones(seq_len, dtype=np.float32) * cat_num
                data, avg[k], std[k] = standardize(data)
                target = raw_data[row_idx, col_idx:col_idx + target_len]
                target -= avg[k]
                target /= std[k]
                target = np.nan_to_num(target)
                target_set[:, k] = np.array(target, dtype=np.float32)
            return [data_set, avg, std, target_set]
        elif self.target_type == 1:
            target_set = np.zeros((target_len, batch_size))
            for k, row_idx in enumerate(row_idxes):
                agent_num, access_num, cat_num = page_info_to_num(self.page_info[row_idx])
                data_set[:, k, 0] = np.array(pro_data[row_idx, :col_idx], dtype=np.float32)
                data_set[:, k, 1] = np.array(self.weekdays, dtype=np.float32)
                data_set[:, k, 2] = np.ones(seq_len, dtype=np.float32) * agent_num
                data_set[:, k, 3] = np.ones(seq_len, dtype=np.float32) * access_num
                data_set[:, k, 4] = np.ones(seq_len, dtype=np.float32) * cat_num

                target = raw_data[row_idx, col_idx:col_idx + target_len]
                #target = np.nan_to_num(target)
                target_set[:, k] = np.array(target, dtype=np.float32)
            return [data_set, avg, std, target_set]



class DataPro(object):
    def __init__(self, data_pro_pars=None):
        self.raw_data_file = data_pro_pars['raw_data_file']
        self.key_file = '../input/key_{}.csv'.format(self.raw_data_file[-5])
        #key_dict = gen_key_dict(self.key_file)
        self.load_len = data_pro_pars['load_len']
        self.batch_size = data_pro_pars['batch_size']
        self.k_fold = data_pro_pars['k_fold']
        self.col_start = data_pro_pars['col_start']
        #self.col_end = data_pro_pars['col_end']
        #self.col_idxes = list(range(self.col_start, self.col_end))
        self.target_type = data_pro_pars['target_type']
        self.n_features = data_pro_pars['n_features']
        self.shuffle_seed = data_pro_pars['shuffle_seed']
        self.init_data()

    def init_data(self):
        '''
        if os.pathisfile(df_file):
            train_data = load_dump(df_file)
        else:
            train_data = load_data(self.raw_data_file, load_len=self.load_len)
            train_data = merge_update_data(train_data)
        '''
        train_data = load_data(self.raw_data_file, load_len=self.load_len)
        #train_data = merge_update_data(train_data)

        #df = df.sample(frac=1).reset_index(drop=True)
        train_data = sklearn.utils.shuffle(train_data, random_state=9527)
        data_df, self.page_info = pre_process(train_data)
        self.data_np = np.asarray(data_df.values, dtype=np.float32)
        self.col_end = self.data_np.shape[1] - 1
        self.col_idxes = list(range(self.col_start, self.col_end))
        self.col_idxes.extend(list(range(430, 440)))

        if self.k_fold > 1:
            self.train_row_idxes, self.validate_row_idxes = gen_k_fold(self.data_np, self.k_fold)
        else:
            self.train_row_idxes = [list(range(self.data_np.shape[0]))]
            self.validate_row_idxes = [[0]]

        return self.data_np, self.train_row_idxes, self.validate_row_idxes, self.page_info

    def sel_batch_builder(self, k=0):
        assert k < self.k_fold
        if self.n_features > 1:
            self.train_bb = BatchBuilder2DMF(self.data_np, self.train_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np,
                                             batch_size=self.batch_size, target_type=self.target_type, page_info=self.page_info, pro_mode='train')
            self.validate_bb = BatchBuilder2DMF(self.data_np, self.validate_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np,
                                                batch_size=self.batch_size, target_type=self.target_type, page_info=self.page_info, pro_mode='validate')
        elif self.n_features == 1:
            self.train_bb = BatchBuilder2D(self.data_np, self.train_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np, batch_size=self.batch_size, target_type=self.target_type)
            self.validate_bb = BatchBuilder2D(self.data_np, self.validate_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np, batch_size=self.batch_size, target_type=self.target_type)
        return self.train_bb, self.validate_bb

    def predict_batch_builder(self):
        predict_row_idxes = list(range(len(self.data_np)))
        if self.n_features > 1:
            self.predict_bb = BatchBuilder2DMF(self.data_np_pro, predict_row_idxes, col_idxes=[self.data_np.shape[-1]], raw_data=self.data_np,
                                               batch_size=self.batch_size, shuffle=False, drop_last=False, target_type=0, page_info=self.page_info, pro_mode='predict')
        elif self.n_features == 1:
            self.predict_bb = BatchBuilder2D(self.data_np, predict_row_idxes, col_idxes=[self.data_np.shape[-1]], raw_data=self.data_np,
                                                batch_size=self.batch_size, shuffle=False, drop_last=False, target_type=0)

    @staticmethod
    def torch_transform(x):
        return torch.log(x + 1.0)

    @staticmethod
    def torch_inv_transform(x):
        return torch.exp(x) - 1.0


class DataProLog(DataPro):
    def __init__(self, data_pro_pars=None):
        super().__init__(data_pro_pars)

    def init_data(self):
        super().init_data()
        self.data_np_pro = self.np_transfrom(self.data_np)

    def np_transfrom(self, x):
        #y = np.log(x + 1.0)
        y = np.log1p(x)
        return y

    @staticmethod
    def torch_transform(x):
        #y = torch.log(x + 1.0)
        #return y
        #to avoid the error
        #RuntimeError: in-place operations can be only used on variables that don't share storage with
        # any other variables, but detected that there are 2 objects sharing it
        y = x.clone()
        return y

    @staticmethod
    def torch_inv_transform(x):
        y = torch.exp(x) - 1.0
        return y

    def sel_batch_builder(self, k=0):
        assert k < self.k_fold
        if self.n_features > 1:
            self.train_bb = BatchBuilder2DMF(self.data_np_pro, self.train_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np,
                                             batch_size=self.batch_size, target_type=self.target_type, page_info=self.page_info, pro_mode='train')
            self.validate_bb = BatchBuilder2DMF(self.data_np_pro, self.validate_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np,
                                                batch_size=self.batch_size, target_type=self.target_type, page_info=self.page_info, pro_mode='validate')
        elif self.n_features == 1:
            self.train_bb = BatchBuilder2D(self.data_np_pro, self.train_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np, batch_size=self.batch_size, target_type=self.target_type)
            self.validate_bb = BatchBuilder2D(self.data_np_pro, self.validate_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np, batch_size=self.batch_size, target_type=self.target_type)
        return self.train_bb, self.validate_bb

    def predict_batch_builder(self):
        predict_row_idxes = list(range(len(self.data_np)))
        if self.n_features > 1:
            self.predict_bb = BatchBuilder2DMF(self.data_np_pro, predict_row_idxes, col_idxes=[self.data_np.shape[-1]], raw_data=self.data_np,
                                                batch_size=self.batch_size, shuffle=False, drop_last=False, target_type=0, page_info=self.page_info, pro_mode='predict')
        elif self.n_features == 1:
            self.predict_bb = BatchBuilder2D(self.data_np_pro, predict_row_idxes, col_idxes=[self.data_np.shape[-1]], raw_data=self.data_np,
                                                batch_size=self.batch_size, shuffle=False, drop_last=False, target_type=0)


class DataProMedian(DataPro):
    def __init__(self, data_pro_pars=None):
        super().__init__(data_pro_pars)

    def init_data(self):
        super().init_data()
        #self.data_np_pro = self.np_transfrom(self.data_np)

    def np_transfrom(self, x, median_len=365):
        self.median = np.zeros_like(x)
        for rk in range(x.shape[0]):
            for ck in range(self.col_start, self.col_end):
                start = max(ck - median_len, 0)
                self.median[rk][ck] = np.median(x[rk][start:ck+1])

        y = np.log(x - self.median + 1.0)
        return y

    def torch_transform(self, x):
        self.median, _ = torch.median(x[:, :, 0], 0)
        #y = torch.log(x[:, :, 0] - self.median + 1.0).unsqueeze(2)
        y = (torch.log(x[:, :, 0] + 1.0) - torch.log(self.median + 1.0)).unsqueeze(2)
        return y

    def torch_inv_transform(self, x):
        #y = (torch.exp(x[:, :, 0]) + self.median - 1.0).unsqueeze(2)
        y = torch.exp(x) * (self.median + 1.0) - 1.0
        return y

    def sel_batch_builder(self, k=0):
        assert k < self.k_fold
        if self.n_features > 1:
            self.train_bb = BatchBuilder2DMF(self.data_np, self.train_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np,
                                             batch_size=self.batch_size, target_type=self.target_type, page_info=self.page_info, pro_mode='train')
            self.validate_bb = BatchBuilder2DMF(self.data_np, self.validate_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np,
                                                batch_size=self.batch_size, target_type=self.target_type, page_info=self.page_info, pro_mode='validate')
        elif self.n_features == 1:
            self.train_bb = BatchBuilder2D(self.data_np, self.train_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np, batch_size=self.batch_size, target_type=self.target_type)
            self.validate_bb = BatchBuilder2D(self.data_np, self.validate_row_idxes[k], col_idxes=self.col_idxes, raw_data=self.data_np, batch_size=self.batch_size, target_type=self.target_type)
        return self.train_bb, self.validate_bb

    def predict_batch_builder(self):
        predict_row_idxes = list(range(len(self.data_np)))
        if self.n_features > 1:
            self.predict_bb = BatchBuilder2DMF(self.data_np_pro, predict_row_idxes, col_idxes=[self.data_np.shape[-1]], raw_data=self.data_np,
                                               batch_size=self.batch_size, shuffle=False, drop_last=False, target_type=0, page_info=self.page_info, pro_mode='predict')
        elif self.n_features == 1:
            self.predict_bb = BatchBuilder2D(self.data_np, predict_row_idxes, col_idxes=[self.data_np.shape[-1]], raw_data=self.data_np,
                                                batch_size=self.batch_size, shuffle=False, drop_last=False, target_type=0)


