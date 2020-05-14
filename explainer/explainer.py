from utils.data_processor import DataLoader_TS
import numpy as np
import pandas as pd
from copy import deepcopy
from utils.timer import Timer
#from utils.mutiprocess import MultiProcess
from explainer.optimizer import myOptimizer
from explainer.valid_exp import Valid_test
import keras.backend as K
import keras
from keras.layers import Layer,Input,Dense
from keras.optimizers import SGD


def TStoDF(TSs:np.array,cols):
    '''
    时间序列数据集转化为pd.DataFrame,并去重
    :param TSs:
    :return:
    '''
    res=pd.DataFrame(TSs.reshape(TSs.shape[0]*TSs.shape[1],TSs.shape[2]),columns=cols)
    res.drop_duplicates(inplace=True)
    return res

class Explainer:
    def __init__(self, dataloader: DataLoader_TS, data_analyzer,engine, exp_config,test_engine:Valid_test=None):
        self.dataloader = dataloader
        self.data_analyzer = data_analyzer
        #self.index_to_repair = [x + 1 for x in dataloader.feature_cols]
        self.exp_config = exp_config
        #self.ignored_features = [x + 1 for x in dataloader.label_cols] + [0]
        if self.exp_config.get('all_data', False):
            self.ori_x, self.ori_y = dataloader.get_all_data()
        else:
            self.ori_x, self.ori_y = dataloader.get_test_data()
        self.engine=engine
        self.length=self.exp_config.get('length', 1)
        self.test_engine=None
        self.valid=False
        if exp_config.get("valid",False) and test_engine is not None:
            self.test_engine=test_engine
            self.valid=True

    def interpolation(self,data,feature):
        col_idx = self.dataloader.features.index(feature)
        source = data.copy()
        target = data.copy()
        target[-1,col_idx] = data[0,col_idx]
        target[:-1,col_idx] = data[1:,col_idx]
        timer = Timer()
        timer.start()
        fixed = self.engine.fix(source, feature, deepcopy(target))
        # fixed=target
        # fixed=np.array(fixed)
        timer.stop()
        error = abs(target[:, col_idx] - fixed[:, col_idx])
        error = error.mean()
        dis = abs(source[:, col_idx] - fixed[:, col_idx])
        dis = dis.mean()
        return fixed, error, dis

    def interpolation_seq(self, data_x, feature, pos):
        source=[]
        target=[]
        ori_shape=data_x.shape
        col_idx=self.dataloader.features.index(feature)
        for mts in data_x:
            for i in range(self.length):
                source.append(mts[pos+i].copy())
                tmp=mts[pos+i].copy()
                tmp[col_idx]=mts[pos-1][col_idx] if pos>0 else mts[pos][col_idx]
                #tmp[col_idx]=min(mts[:,col_idx])
                target.append(tmp)
                #fixed.append(self.fix(source[-1],col_idx,tmp.copy()))
        source=np.array(source)
        target=np.array(target)
        timer=Timer()
        timer.start()
        fixed=self.engine.fix(source,feature,deepcopy(target))
        #fixed=target
        #fixed=np.array(fixed)
        timer.stop()
        error=abs(target[:,col_idx]-fixed[:,col_idx])
        error=error.mean()
        dis=abs(source[:,col_idx]-fixed[:,col_idx])
        dis=dis.mean()
        fixed=fixed.reshape((ori_shape[0],self.length,ori_shape[2]))
        for idx,fix in enumerate(fixed):
            for i in range(self.length):
                data_x[idx][pos+i]=fix[i]
        return data_x,error,dis

    def valid_test(self,new_data:np.array,feature):
        new_accs,ori_acc=self.test_engine.valid(TStoDF(new_data,self.dataloader.features),feature)
        return new_accs,ori_acc

    def cal_FI(self,feature):
        res={}
        feature_acc={}
        ori_acc=0
        x,target_dis,source_dis=self.interpolation(self.ori_x.copy(), feature)
        if self.valid:
            new_accs,ori_acc=self.valid_test(x,feature)
            for k,v in new_accs.items():
                    feature_acc[k]=v
        pre_y=self.data_analyzer.predict_point_by_point(x)
        # if (pre_y==self.ori_y).all():
        #     print("=======Same===========")
        pre_y=np.array([1 if x>0.5 else 0 for x in pre_y])
        res['tot']=len(pre_y[pre_y!=self.ori_y[:,0]])/len(pre_y)
        res['threshold']=self.engine.threshold[feature]
        res['ori_acc']=ori_acc
        res['target_dis']=target_dis
        res['source_dis']=source_dis
        res["sum_dis"]=0
        for k in feature_acc.keys():
            res[k] = feature_acc[k]
            res["sum_dis"]+=res[k]
        res['this']=res[feature]
        return res

    def cal_FI_seq(self, feature):
        res={}
        n=0
        feature_acc={}
        ori_acc=0
        target_dis=0
        source_dis=0
        for i in range(0,self.dataloader.seq_len,self.exp_config.get('length',1)):
            n+=1
            #if i==0:#0的结果和其他相差太大
            #    continue
            x,tmp,tmp2=self.interpolation_seq(self.ori_x.copy(), feature, i)
            target_dis += tmp
            source_dis+=tmp2
            if self.valid:
                new_accs,tmp=self.valid_test(x,feature)
                ori_acc+=tmp
                for k,v in new_accs.items():
                    if k in feature_acc:
                        feature_acc[k]+=v
                    else:
                        feature_acc[k]=v
            pre_y=self.data_analyzer.predict_point_by_point(x)
            # if (pre_y==self.ori_y).all():
            #     print("=======Same===========")
            pre_y=np.array([1 if x>0.5 else 0 for x in pre_y])
            res[str(i).zfill(3)]=len(pre_y[pre_y!=self.ori_y[:,0]])/len(pre_y)

        res['tot']=np.sum(list(res.values()))#此处res里应该都是重要度
        res['threshold']=self.engine.threshold[feature]
        res['ori_acc']=ori_acc/n
        res['target_dis']=target_dis/n
        res['source_dis']=source_dis/n
        res["sum_dis"]=0
        for k in feature_acc.keys():
            res[k] = feature_acc[k]/n
            res["sum_dis"]+=res[k]
        res['this']=res[feature]
        return res
