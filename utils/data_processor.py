import time
import random
import numpy as np
import pandas as pd
from copy import deepcopy


def get_index(dataframe, targets):
    result = []
    for feature in targets:
        if feature in dataframe.columns:
            result.append(dataframe.columns.get_loc(feature))
    return result


class DataLoader:
    """一般数据"""
    def __init__(self, dataframe, features, labels, valid_ratio, test_ratio,discrete=[], normalise=False,
                 shuffle=True, balance=True, add_rand=False, add_label=False):
        """

        :param dataframe:
        :param features:
        :param labels:
        :param valid_ratio: 验证集比例
        :param test_ratio: 测试集比例
        :param normalise: 是否z-score
        :param shuffle: 是否打乱
        :param balance: 是否对标签进行类别平衡
        :param add_rand:
        :param add_label:
        :param discrete:离散数据变量列表
        """
        self.add_label = add_label
        self.normalise = normalise
        self.shuffle = shuffle
        self.balance = balance if labels else False
        self.add_rand = add_rand

        self.features = deepcopy(features)
        self.labels = deepcopy(labels)
        self.discrete=discrete
        self.discrete_map={}#{编号：原值}
        dataframe = self._init_data(dataframe)
        self.data = dataframe.get(self.features)
        for idx,label in enumerate(self.labels):
            self.data[label+"_label"]=dataframe.get(label)
            self.labels[idx]=label+'_label'
            if self.add_label:
                self.data[label]=dataframe.get(label)
                self.features.append(label)

        self.feature_cols = get_index(self.data, self.features)
        self.label_cols = get_index(self.data, self.labels)

        self.start_stamps = self._gen_stamps(dataframe)
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.split_data()

        self.means = {}
        self.stds = {}
        self.statis()

    def _init_data(self, dataframe:pd.DataFrame):
        if self.add_rand:
            self.features.append('Random_number')
            dataframe.insert(0, 'Random_number', np.random.random(size=len(dataframe)))
        cols = self.features + self.labels
        cols=list(set(cols))
        #for col in self.discrete:
        #    items=dataframe[col].drop_duplicates()
        return dataframe.get(cols).dropna()

    def is_discrete(self,col):
        if self.data[col].max() - self.data[col].min() + 1 == self.data[col].nunique():
            return True
        return False

    """
    def statis(self):
        '''
        应划分后 在训练集上统计
        :return:
        '''
        data_x,_=self.get_train_data()
        data_x=data_x.reshape((data_x.size//len(self.features),len(self.features)))
        for idx,col in enumerate(self.features):
            if self.data[col].max() - self.data[col].min() + 1 == self.data[col].nunique():
                continue
            else:
                self.means[col] = data_x[:,idx].mean()
                self.stds[col] = data_x[:,idx].std() / self.STD_SCALE
        if self.normalise:
            for col in self.means.keys():
                self.data[col] = (self.data[col] - self.means[col]) / self.stds[col]
    """

    def statis(self):
        for col in self.features:
            if self.data[col].max() - self.data[col].min() + 1 == self.data[col].nunique():
                continue
            else:
                self.means[col] = self.data[col].mean()
                self.stds[col] = self.data[col].std()
        if self.normalise:
            for col in self.means.keys():
                self.data[col] = (self.data[col] - self.means[col]) / self.stds[col]

    def single_score(self, input):
        new_input = input.copy()
        if len(input.shape)==1:
            new_input=new_input.reshape((1,input.size))
        for col in self.means.keys():
            idx = self.data.columns.get_loc(col)
            new_input[:,idx] = (input[:,idx] - self.means[col]) / self.stds[col]
        return new_input.reshape(input.shape)

    def single_descore(self, input):
        new_input = input.copy()
        if len(input.shape)==1:
            new_input=new_input.reshape((1,input.size))
        for col in self.means.keys():
            idx = self.data.columns.get_loc(col)
            new_input[:,idx] = input[:,idx] * self.stds[col] + self.means[col]
        return new_input.reshape(input.shape)

    def de_zscore(self):
        if self.normalise:
            for col in self.means.keys():
                self.data[col] = self.data[col] * self.stds[col] + self.means[col]

    def set_ratio(self,valid_ratio , test_ratio):
        ratio1 = int(len(self.start_stamps) * (1 - valid_ratio - test_ratio))
        ratio2 = ratio1 + int(len(self.start_stamps) * valid_ratio)
        self.train_stamps = self.start_stamps[:ratio1]
        self.valid_stamps = self.start_stamps[ratio1:ratio2]
        self.test_stamps = self.start_stamps[ratio2:]
        self.len_train = len(self.train_stamps)
        self.len_valid = len(self.valid_stamps)
        self.len_test = len(self.test_stamps)

    def split_data(self):
        if self.shuffle:
            random.seed(int(time.time()))
            random.shuffle(self.start_stamps)
        self.set_ratio(self.valid_ratio,self.test_ratio)

    def _gen_stamps(self, dateframe):
        start_stamps = list(self.data.index)
        if self.balance:
            class_num = {}
            for stamp in start_stamps:
                cla = self.data.loc[stamp, self.labels][0]
                class_num[cla] = class_num.get(cla, 0) + 1
            maxn = max(class_num.values())
            new_stamps = []
            random.seed(int(time.time()))
            for stamp in start_stamps:
                cla = self.data.loc[stamp, self.labels][0]
                add_times = int(maxn / class_num[cla])
                new_stamps.extend([stamp] * add_times)
                if random.random() <= (maxn / class_num[cla] - add_times):
                    new_stamps.append(stamp)
            start_stamps = new_stamps
        return start_stamps

    def _next_window(self, mask):
        x = self.data.loc[mask, self.features]
        y = self.data.loc[mask, self.labels]
        return np.array(x), np.array(y)

    def drop_features(self, features=[], change_data=False):
        for feature in features:
            if feature in self.features:
                self.features.remove(feature)
                if change_data:
                    self.data.drop(columns=[feature], inplace=True)
                index = get_index(self.data, [feature])[0]
                self.feature_cols.remove(index)

    def get_data(self, stamps):
        res_x ,res_y=self._next_window(stamps)
        return np.array(res_x), np.array(res_y)

    def get_all_data(self):
        return self.get_data(self.start_stamps)

    def get_test_data(self):
        return self.get_data(self.test_stamps)

    def get_valid_data(self):
        return self.get_data(self.valid_stamps)

    def get_train_data(self):
        return self.get_data(self.train_stamps)

    def generate_train_batch(self, batch_size):
        start = 0
        if batch_size>self.len_train:
            import warnings
            warnings.warn('batch_size>len_train,batch_size too big')
        while True:
            mask=[]
            while(len(mask)!=batch_size):
                remain=batch_size-len(mask)
                if start+remain<self.len_train:
                    mask.extend(self.train_stamps[start:start+remain])
                    start+=remain
                else:
                    mask.extend(self.train_stamps[start:])
                    start=0
            x_batch,y_batch=self._next_window(mask)
            '''
            for b in range(batch_size):
                if i >= len(self.train_stamps):
                    i = 0
                    random.shuffle(self.train_stamps)
                x, y = self._next_window(self.train_stamps[i])
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            '''
            yield np.array(x_batch), np.array(y_batch)

    def get_config(self):
        return {"features":self.features, "labels":self.labels,
                "valid_ratio":self.valid_ratio, 'test_ratio':self.test_ratio,
                'normalise':self.normalise,'shuffle':self.shuffle, 'balance':self.balance,
                'add_rand':self.add_rand, 'add_label':self.add_label}

class DataLoader_TS(DataLoader):
    """时间序列数据"""

    def __init__(self, dataframe, features, labels, time_cols, seq_len, pre_len, valid_ratio, test_ratio=0,
                 normalise=False, shuffle=True, balance=True, add_rand=False, add_label=False, difference=False, duplicate=False, orderspt=False):
        """

        :param dataframe:
        :param features:
        :param labels:
        :param time_cols: 时间列
        :param seq_len: 输入序列长度
        :param pre_len: 预测点距离输入的距离
        :param valid_ratio:
        :param test_ratio:
        :param normalise:
        :param shuffle:
        :param balance:
        :param add_rand:
        :param add_label:
        :param difference: 是否做差分
        :param duplicate: 是否重采样
        :param orderspt: 测试验证集是否按顺序切分
        """
        self.orderspt=orderspt
        self.duplicate=duplicate
        self.difference = difference
        self.pre_len = pre_len
        self.seq_len = seq_len
        self.time_cols = time_cols
        super(DataLoader_TS, self).__init__(dataframe.copy(),
                                            features=features,
                                            labels=labels,
                                            valid_ratio=valid_ratio,
                                            test_ratio=test_ratio,
                                            normalise=normalise,
                                            shuffle=shuffle,
                                            balance=balance,
                                            add_rand=add_rand,
                                            add_label=add_label, )

    def _init_data(self, dataframe):
        if self.add_rand:
            self.features.append('Random_number')
            dataframe.insert(0, 'Random_number', np.random.random(size=len(dataframe)))
        cols = self.features + self.labels
        cols=list(set(cols))
        return dataframe.get(cols + self.time_cols).dropna()

    def split_data(self):
        if self.shuffle and not self.orderspt:
            random.seed(int(time.time()))
            random.shuffle(self.start_stamps)
        self.set_ratio(self.valid_ratio,self.test_ratio)
        if self.shuffle:
            random.seed(int(time.time()))
            random.shuffle(self.train_stamps)

    def _get_consecutive_series(self, dataframe: pd.DataFrame):
        import datetime
        if self.time_cols:
            time_col = self.time_cols[0] if type(self.time_cols) == list and len(
                self.time_cols) == 1 else self.time_cols
            data = dataframe.get(time_col)
            if type(data) == pd.DataFrame:#分散表示
                full = {'year': 0, 'month': 3, 'day': 6, 'hour': 9, 'minute': 12, 'second': 15}
                tmp = []
                for time_col in self.time_cols:
                    low = time_col.lower()
                    if low in full.keys():
                        tmp.append((full[low], time_col))
                tmp.sort()
                pattern1 = "{}-{}-{} {}:{}:{}"[tmp[0][0]:tmp[-1][0] + 2]
                pattern2 = "%Y-%m-%d %H:%M:%s"[tmp[0][0]:tmp[-1][0] + 2]
                sh = data.apply(
                    lambda x: datetime.datetime.strptime(pattern1.format(*[int(x.get(i[1])) for i in tmp]), pattern2),
                    axis=1)
            elif str(data.dtype) != 'int64' and str(data.dtype) != 'float64':
                sh = data.apply(lambda x: datetime.datetime.strptime(x['sh'], "%Y-%m-%d %H:%M:%S"), axis=1)
            else:
                sh = None
            if sh is not None:
                interval_hours = sh[1] - sh[0]
                for idx in range(2, len(sh)):
                    if sh[idx] - sh[idx - 1] == interval_hours:
                        break
                    else:
                        interval_hours = sh[idx] - sh[idx - 1]
                mask = sh.shift(int(-(self.seq_len + self.pre_len - 1))) == sh.apply(
                    lambda x: x + (self.seq_len + self.pre_len - 1) * interval_hours)
                start_stamps = sh[mask].index.tolist()
            else:
                mask = data.shift(int(-(self.seq_len + self.pre_len - 1))) == data.apply(
                    lambda x: x + (self.seq_len + self.pre_len - 1))
                start_stamps = data[mask].index.tolist()
            return start_stamps
        else:
            return self.data.index.tolist()[:-(self.seq_len + self.pre_len) + 1]

    def _gen_stamps(self, data):
        start_stamps = self._get_consecutive_series(data)
        if self.duplicate:
            start_stamps.sort()
            new_stamps=[start_stamps[0]]
            last_stamp=start_stamps[0]
            for stamp in start_stamps:
                if stamp>=last_stamp+self.seq_len:
                    new_stamps.append(stamp)
                    last_stamp=stamp
            start_stamps=new_stamps
        if self.balance:
            class_num = {}
            maxn = 0
            for stamp in start_stamps:
                cla = self.data.loc[stamp + self.seq_len + self.pre_len - 1, self.labels][0]
                class_num[cla] = class_num.get(cla, 0) + 1
                maxn = max(maxn, class_num[cla])
            new_stamps = []
            random.seed(int(time.time()))
            for stamp in start_stamps:
                cla = self.data.loc[stamp + self.seq_len + self.pre_len - 1, self.labels][0]
                add_times = int(maxn / class_num[cla])
                for _ in range(add_times):
                    new_stamps.append(stamp)
                if random.random() <= (maxn / class_num[cla] - add_times):
                    new_stamps.append(stamp)
            start_stamps = new_stamps
        return start_stamps

    def _next_window(self, mask):
        '''Generates the next data window from the given index location mask'''
        res_x,res_y=[],[]
        for i in mask:
            window = self.data.loc[i:i + self.seq_len + self.pre_len-1].values#loc包含头尾
            x = window[:self.seq_len, self.feature_cols]
            x = self.difference_windows(x)[0] if self.difference else x
            y = window[-1, self.label_cols]
            res_x.append(x)
            res_y.append(y)
        return np.array(res_x), np.array(res_y)

    def difference_windows(self, window):
        '''
        对个窗口，每列的第一个值作为基本值，计算该列与基本值的比值-1
        反映了改窗口内值的相对变化程度
        :param window_data:
        :param single_window:
        :return:
        '''
        difference_data = []
        difference_window = []
        for col_i in range(window.shape[1]):
            base = 1
            for p in window[:, col_i]:
                if float(p) != 0:
                    base = p
                break
            difference_col = [((float(p) / float(base))) for p in window[:, col_i]]
            difference_window.append(difference_col)
        difference_window = np.array(
            difference_window).T  # reshape and transpose array back into original multidimensional format
        difference_data.append(difference_window)
        return np.array(difference_data)

    def get_config(self):
        res=super().get_config()
        res.update({
            'orderspt':self.orderspt,
            'duplicate':self.duplicate,
            'difference':self.difference,
            'pre_len':self.pre_len,
            'seq_len' : self.seq_len,
            'time_cols' : self.time_cols})