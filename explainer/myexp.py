from utils.data_processor import DataLoader_TS
from explainer.repairers.GeneralRepairer import Repairer
import pandas as pd
import numpy as np
from copy import deepcopy
from model.DNN import DNN
from config import *

def com_cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return np.NAN
    else:
        return dot_product / ((normA*normB)**0.5)

class myExp:
    def __init__(self, dataloader: DataLoader_TS,data_analyzer,EXP_CONFIG):
        self.dataloader=dataloader
        self.data_analyzer=data_analyzer
        self.data=[]
        for idx,row in dataloader.data.iterrows():
            self.data.append([idx]+row.to_list())#把index作为第0列加入数据
        self.index_to_repair=[x+1 for x in dataloader.feature_cols]
        self.exp_config=EXP_CONFIG
        self.ignored_features=[x+1 for x in dataloader.label_cols]+[0]
        if self.exp_config.get('all_data',False):
            self.ori_x, self.ori_y = dataloader.get_all_data()
        else:
            self.ori_x,self.ori_y=dataloader.get_test_data()

    def repair(self,index):
        repairer = Repairer(self.
                            data.copy(), index+1,
                            self.exp_config.get('repair_level',0.2), False, features_to_ignore=self.ignored_features)
        repaired_data=repairer.repair(self.data.copy())
        df=pd.DataFrame(repaired_data).set_index(0)
        df.columns=self.dataloader.data.columns
        tmp=self.dataloader.data.copy()
        self.dataloader.data=df
        if self.exp_config.get('all_data',False):
            data_x,_=self.dataloader.get_all_data()
        else:
            data_x,_=self.dataloader.get_test_data()
        self.dataloader.data=tmp
        return data_x,df

    def interpolation(self,data_x,col_idx,pos):
        for i in range(len(data_x)):
            avg = data_x[i][:][col_idx].sum() / data_x[i][:][col_idx].size
            for j in range(self.exp_config.get('length',1)):
                data_x[i][pos+j][col_idx] = data_x[i][pos-1][col_idx] if pos!=0 else avg
        return data_x

    def valid_train(self,df,feature):
        '''
        todo:
        :param df:
        :param feature:
        :return:
        '''
        df = df.join(self.dataloader.time_data)
        dataloader = DataLoader_TS(
            dataframe=df,
            seq_len=self.dataloader.seq_len,
            pre_len=self.dataloader.pre_len,
            split=self.dataloader.split,
            features=[x for x in self.dataloader.features if x != feature],
            labels=[feature],
            time_cols=self.dataloader.time_cols,
            shuffle=self.dataloader.shuffle,
            normalise=False,
            balance=False,
            add_rand=False,
            add_label=False,
        )
        val_model_config = VAL_MODULE_CONFIG.copy()
        val_model_config["layers"][0]['input_timesteps'] = dataloader.seq_len
        val_model_config["layers"][0]['input_dim'] = len(dataloader.features)
        val_train_config=VAL_TRAIN_CONFIG.copy()
        val_train_config['callback'][0]["min_delta"]=df[feature].mean()/20
        model = DNN("tmp", dataloader, val_model_config, train_config=val_train_config)
        acc = model.get_model(True)
        return acc

    def valid(self,df:pd.DataFrame,feature):
        '''
        todo: 暂时不验证了
        :param df:
        :param feature:
        :return:
        '''
        return
        old_acc=self.valid_train(self.dataloader.data.copy(),feature)

        df.drop(columns=self.dataloader.labels+[feature],inplace=True)
        df[feature] = self.dataloader.data[feature]
        new_acc=self.valid_train(df,feature)

        similar=0
        for (idx1,x),(idx2,y) in zip(self.dataloader.data.iterrows(),df.iterrows()):
            if idx1==idx2:
                similar+=com_cos(x.drop(feature).values,y.drop(feature).values)
        return new_acc,old_acc,similar/len(df)

    def cal_FI(self,ori,feature_idx):
        res={}
        feature=self.dataloader.features[feature_idx]
        data_x,df = self.repair(feature_idx)
        if self.exp_config.get('valid',False):
            res["new_rate"],res["old_rate"],res['similar']=self.valid(df,feature)
        res['tot'] = ori-self.data_analyzer.model.evaluate(data_x, self.ori_y, batch_size=4096)[1]
        for i in range(0,self.dataloader.seq_len,self.exp_config.get('length',1)):
            if i==0:#0的结果和其他相差太大
                continue
            x=self.interpolation(deepcopy(data_x),feature_idx,i)
            acc=self.data_analyzer.model.evaluate(x,self.ori_y,batch_size=4096)[1]
            # pre_y=self.data_analyzer.predict_point_by_point(x)
            # if (pre_y==self.ori_y).all():
            #     print("=======Same===========")
            res[str(i)]=ori-acc
        return res

if __name__ == '__main__':
    df=pd.DataFrame([[1,2,3],[4,5,6]])
    data=[]
    for col in df:
        data.append(df[col].to_list())
    print(data)