import lightgbm as lgb
import numpy as np
import pandas as df
from utils.data_processor import DataLoader_TS
from model.data_analyzer import DataAnalyzer

def TSto2D(TS:np.array):
    return TS.reshape((TS.shape[0],TS.shape[1]*TS.shape[2]))

class Lgbm(DataAnalyzer):
    def __init__(self,data_processor:DataLoader_TS,config,data_config=None):
        super(Lgbm, self).__init__("GBDT", data_processor, config, config, data_config)
        self.model=None
        self.features=[]
        for idx in range(self.data_processor.seq_len):
            for feature in self.data_processor.features:
                self.features.append("{}_{}".format(feature,idx))

    def test_model(self):
        self.test_x, self.test_y = self.data_processor.get_test_data()
        self.test_x = TSto2D(self.test_x)
        # print(self.test_y.shape)
        pre_y = [[1] if x > 0.5 else [0] for x in self.model.predict(self.test_x)]
        pre_y = np.array(pre_y)
        gain = self.get_importance()
        for col in gain:
            print(col, gain[col].sum())
        return len(pre_y[pre_y == self.test_y]) / len(pre_y)
        
    def train(self):
        train_x, train_y = self.data_processor.get_train_data()
        train_data = lgb.Dataset(TSto2D(train_x), feature_name=self.features, free_raw_data=False)
        train_data.set_label(train_y.reshape((train_y.shape[0],)))
        train_data.construct()

        valid_x, valid_y = self.data_processor.get_valid_data()
        valid_data = lgb.Dataset(TSto2D(valid_x), feature_name=self.features, free_raw_data=False)
        valid_data.set_label(valid_y.reshape((valid_y.shape[0],)))
        valid_data.construct()
        self.model=lgb.train(self.model_config,train_data,valid_sets=[valid_data],valid_names=['valid'])

    def get_importance(self):
        #split=self.model.feature_importance('split')
        gain=self.model.feature_importance('gain')
        #split=split.reshape((self.dataloader.seq_len,len(self.dataloader.features)))
        #split=df.DataFrame(split,columns=self.dataloader.features)
        gain = gain.reshape((self.data_processor.seq_len, len(self.data_processor.features)))
        gain = df.DataFrame(gain, columns=self.data_processor.features)
        return gain

    def predict_point_by_point(self,x):
        predicted=self.model.predict(TSto2D(x))
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
