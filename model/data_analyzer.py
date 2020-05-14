import os
from utils.data_processor import DataLoader
from utils.timer import Timer
import keras
from keras.models import Model
from model.utils import *
import json

def cfg_compare(src_cfg,obj_file):
    with open(obj_file,'r',encoding='utf-8') as fp:
        obj_cfgs=json.load(fp)
    for obj_cfg in obj_cfgs.values():
        if obj_cfg==src_cfg:
            return True
    return False

def find_cgf(save_dir,data_config,model_config):
    dirs = os.listdir(save_dir)
    for file in dirs:
        full_file = os.path.join(save_dir, file)
        if file.endswith(".js") and cfg_compare(data_config, full_file) and cfg_compare(model_config, full_file):
            obj_id = file[:file.rfind('.')]
            return obj_id
    return ""

class DataAnalyzer():
    SAVE_DIR=r".\saved_models"
    def __init__(self,my_id,data_processor:DataLoader,model_config,train_config=None,data_config=None):
        self.model = Model()
        self.data_processor=data_processor
        self.data_config=data_config
        self.model_config=model_config
        self.train_config=train_config
        self.name=self.model_config.get('name', my_id)
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        self.realize=True

    def get_trained(self,by_name=False):
        if by_name or self.model_config.get("by_name",False):
            obj_id=self.name
        else:
            if not self.data_config:
                return False
            obj_id=find_cgf(self.SAVE_DIR,self.data_config,self.model_config)
        if obj_id:
            try:
                self.name = obj_id
                self.load_model(os.path.join(self.SAVE_DIR,self.name + ".h5"))
                return True
            except Exception as e:
                print("[Model] load_model Failed")
                try:
                    self.name=obj_id
                    self.load_model(os.path.join(self.SAVE_DIR,self.name + ".h5"),only_weight=True)
                    return True
                except:
                    #raise
                    print(e)
        return False

    def del_trained(self):
        if not self.data_config:
            return False
        obj_id = find_cgf(self.SAVE_DIR, self.data_config, self.model_config)
        if obj_id:
            try:
                os.remove(os.path.join(self.SAVE_DIR,obj_id + '.js'))
                os.remove(os.path.join(self.SAVE_DIR,obj_id + '.h5'))
            except Exception as e:
                print(e)

    def get_model(self, test=False):
        if(not(self.train_config and self.train_config.get("retrain",True))):
            if(self.get_trained()):
                if test:
                    return self.test_model()
                else:
                    return -1
        if not self.train_config:
            print("No Train Config")
            return -1
        self.del_trained()
        return self.train_model(test)

    def test_model(self):
        if self.data_processor.len_test > 0:
            data_x, data_y = self.data_processor.get_test_data()
            acc = self.model.evaluate(data_x, data_y, batch_size=4096)
            if type(acc)!=list:
                acc=[acc]
            return acc
        else:
            return [-1]

    def train_model(self,test):
        '''
        :param test:
        :return:
        '''
        with open(os.path.join(self.SAVE_DIR, self.name + ".js"), "w", encoding='utf-8') as fp:
            json.dump({"data_config":self.data_config,"train_config":self.model_config},fp,ensure_ascii=False)
        self.build_model()
        self.train_generator()
        if not self.realize:
            self.train()
        if test:
            return self.test_model()
        else:
            return -1

    def load_metrics(self,metrics):
        metric_dic={}
        for metric in metrics:
            if metric=='fmeasure':
                metric_dic['fmeasure']=fmeasure
            elif metric=='recall':
                metric_dic['recall']=recall
            elif metric=='precision':
                metric_dic['precision']=precision
            else:
                metric_dic[metric]=metric
        return metric_dic

    def load_model(self, filepath,only_weight=False):
        print('[Model] Loading model from file %s' % filepath)
        if only_weight:
            self.build_model()
            self.model.load_weights(filepath)
        else:
            self.model = keras.models.load_model(filepath)
        print("[Model] load success")

    def build_model(self):
        pass

    def train(self):
        pass

    def train_generator(self):
        self.realize=False

    def predict_point_by_point(self, data):
        pass