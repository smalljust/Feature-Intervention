from utils.data_processor import DataLoader
from model.DNN import DNN
import pandas as pd
from copy import deepcopy
import numpy as np

def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton

@Singleton
class Valid_test():
    def __init__(self,data:pd.DataFrame,data_config,valid_model_config,valid_train_config):
        self.data=data
        self.data_config=data_config
        if self.data_config['add_rand']:
            if 'Random_number'not in self.data_config['features']:
                self.data_config['features'].append('Random_number')
            if 'Random_number' not in self.data.columns:
                self.data.insert(0, 'Random_number', np.random.random(size=len(self.data)))
            self.data_config['add_rand']=False
        self.valid_model_config=valid_model_config
        self.valid_train_config=valid_train_config
        self.model_builders={}
        self.acc={}

    def get_DA(self, feature):
        if feature in self.model_builders.keys():
            return self.model_builders[feature]
        my_data_config = deepcopy(self.data_config)
        # 为多标签情况做准备
        features=[feature]
        my_data_config['features'] = [f for f in my_data_config["features"] if f not in features]
        my_data_config['labels'] = features
        my_data_config['balance'] = False
        my_data_config['add_rand'] = False
        my_data_config['add_label'] = False
        data_loader = DataLoader(
            deepcopy(self.data),
            features=my_data_config['features'],
            labels=my_data_config['labels'],
            valid_ratio=my_data_config['valid_ratio'],
            test_ratio=my_data_config['test_ratio'],
            normalise=my_data_config['normalise'],
            shuffle=my_data_config['shuffle'],
            balance=my_data_config['balance'],
            add_rand=my_data_config['add_rand'],
            add_label=my_data_config['add_label'],
        )
        if my_data_config['normalise']:
            mean = data_loader.data[my_data_config['labels'][0]].mean()
            std = (data_loader.data[my_data_config['labels'][0]].std())
            data_loader.data[my_data_config['labels'][0]] = (data_loader.data[
                                                                 my_data_config['labels'][0]] - mean) / std
        model_builder = DNN(str(my_data_config['labels'][0]), data_loader, self.valid_model_config,
                            self.valid_train_config, my_data_config)
        if model_builder.get_trained(by_name=True):
            return model_builder
        model_builder.train_model(False)
        return model_builder

    def get_all_model_builder(self):
        for feature in self.data_config['features']:
            self.model_builders[feature]=self.get_DA(feature)
            self.acc[feature]=self.model_builders[feature].test_model()[-1]

    def valid(self,new_data:pd.DataFrame,feature):
        if not self.model_builders:
            self.get_all_model_builder()
        new_accs={}
        for now_feature in new_data.columns:
            model_builder=self.model_builders[feature]
            labels=new_data[now_feature]
            features=new_data[[f for f in new_data.columns if f != now_feature]]
            new_accs[now_feature]=model_builder.model.evaluate(features,labels,verbose=0)[1]
        return new_accs,self.acc[feature]
