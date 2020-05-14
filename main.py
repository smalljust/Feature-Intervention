from utils.data_processor import DataLoader_TS
from utils.plot import plot_feature,plot_dis
from config import *
from model.DNN import DNN
import os
import datetime as dt
from model.vae import *
import json
from explainer.explainer import Explainer
from explainer.valid_exp import Valid_test
import copy
from explainer.VAE_engine import VAE_engine
from model.GBDT import Lgbm
from model.gan import GAN

def get_dataloader(TS:bool,config=None,df=None):
    timer = Timer()
    timer.start()
    my_data_config = copy.deepcopy(config if config else DATA_CONFIG)
    if df is None:
        df = pd.read_csv(open(os.path.join('data', my_data_config['filename'])))
    if TS:
        data_loader = DataLoader_TS(
            dataframe=df,
            features=my_data_config['features'],
            labels=my_data_config['labels'],
            time_cols=my_data_config['time_cols'],
            seq_len=my_data_config['sequence_length'],
            pre_len=my_data_config['predict_length'],
            valid_ratio=my_data_config['valid_ratio'],
            test_ratio=my_data_config['test_ratio'],
            normalise=my_data_config['normalise'],
            shuffle=my_data_config['shuffle'],
            balance=my_data_config['balance'],
            add_rand=my_data_config['add_rand'],
            add_label=my_data_config['add_label'],
            difference=my_data_config['difference'],
        )
    else:
        data_loader = DataLoader(
            dataframe=df,
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
    print('[DataLoader] Data Loaded')
    timer.stop()
    return data_loader, my_data_config

def get_valid_test(data_config):
    df = pd.read_csv(open(os.path.join('data', data_config['filename'])))
    return Valid_test(df,copy.deepcopy(data_config),VAL_MODULE_CONFIG,VAL_TRAIN_CONFIG)

def cal_FI(myid, data_loader_ts, model_builder, acc):
    res_dir = "./res/" + myid
    os.makedirs(res_dir)
    my_data_config2 = copy.deepcopy(DATA_CONFIG)
    my_data_config2['labels']=[]
    '''
    把真实的label替换成预测的label
    df = pd.read_csv(open(os.path.join('data', my_data_config2['filename'])))
    data_x, _ = data_loader_ts.get_all_data()
    data_y = model_builder.predict_point_by_point(data_x)
    df[my_data_config2['labels'][0]] = np.nan
    for idx, i in enumerate(data_loader_ts.start_stamps):
        df.loc[i + data_loader_ts.seq_len + data_loader_ts.pre_len - 1, my_data_config2['labels'][0]] = 1 if data_y[idx] > 0.5 else 0
    '''
    data_loader,_=get_dataloader(False,my_data_config2)
    explainer = Explainer(data_loader_ts, model_builder, get_engine(data_loader,my_data_config2,EXP_CONFIG), EXP_CONFIG,
                          get_valid_test(DATA_CONFIG))
    EXP_CONFIG['name'] = model_builder.name
    EXP_CONFIG['acc'] = acc
    EXP_CONFIG['rec_loss']=explainer.engine.rec_loss
    EXP_CONFIG['KL_loss'] = explainer.engine.KL_loss
    res_df = pd.DataFrame()
    source_dis={}
    target_dis={}
    for idx, f in enumerate(data_loader_ts.features):
        ans = explainer.cal_FI_seq(f)
        with open(res_dir + '/' + str(f) + ".txt", "w") as fp:
            json.dump(ans, fp)
        ans["name"] = str(f)
        source_dis[str(f)]=ans["source_dis"]
        target_dis[str(f)]=ans["target_dis"]
        res_df = res_df.append(ans, ignore_index=True)
        plot_feature(ans, res_dir + '/' + str(f))
    res_df.to_csv(res_dir + r"/ans.csv", index=False)
    plot_dis(source_dis,target_dis,res_dir + '/ans')
    with open(res_dir + r'/config.js', 'w') as fp:
        EXP_CONFIG['VAE_MODEL_CONFIG']=VAE_MODEL_CONFIG
        EXP_CONFIG['VAE_TRAIN_CONFIG']=VAE_TRAIN_CONFIG
        json.dump(EXP_CONFIG, fp)

def get_DNN(myid, data_loader_ts, my_data_config):
    model_builder = DNN(myid, data_loader_ts, MODULE_CONFIG, TRAIN_CONFIG, my_data_config)
    acc=model_builder.get_model(True)
    return model_builder, acc

def get_engine(data_loader:DataLoader,data_config,exp_config):
    return VAE_engine(data_loader,data_config,exp_config)

def get_vae(data_loader:DataLoader):
    generator=VAE(data_loader, VAE_MODEL_CONFIG, VAE_TRAIN_CONFIG)
    generator.get_model()
    generator.cal_threshold()
    #generator.lr()
    #if VAE_MODEL_CONFIG["latent_dim"]==2:
    #    show_latent_space(data_loader.get_test_data()[0],generator.encoder,generator.linears)
    return generator

def get_GBDT(myid,data_loader_ts,my_data_config):
    model = Lgbm(data_loader_ts, {
        "objective": "binary",
        "num_iteration": 100,
        "learning_rate": 0.1,
        'early_stopping_round':5
    })
    acc=model.get_model(True)
    return model,acc

def get_GAN(myid,data_loader,my_data_config):
    model=GAN(data_loader, {"latent_dim":10,"hidden_dim":10},{"epoch":10000,"batch_size":256})
    model.get_model()
    return model,None

def fix_test():
    my_data_config2 = copy.deepcopy(DATA_CONFIG)
    my_data_config2['labels'] = []
    data_loader, _ = get_dataloader(False, my_data_config2)
    engine = get_engine(data_loader, my_data_config2,EXP_CONFIG)
    #print(engine.threshold)
    a = 8848
    b = 8831
    df = data_loader.data.loc[b: a]
    #print(df)
    target=df.copy()
    target.iloc[:,0]=-1.0
    print(target.values)
    print(engine.fix(df.values.copy(),'Temp', target.values.copy()))
    """
    temp=DNN("",None,VAL_MODULE_CONFIG)
    temp.build_model()
    temp.load_model("saved_models/Temp.h5")
    print(temp.model.evaluate(df.iloc[:,1:].values,target.iloc[:,0].values))
    """

def exp():
    myid = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    dlts,dlts_cfg=get_dataloader(True)
    model, acc=get_DNN(myid, dlts, dlts_cfg)
    cal_FI(myid,dlts,model,acc)

def test_model():
    my_data_config = copy.deepcopy(DATA_CONFIG)
    my_data_config['balance']=False
    dlts,dlts_config=get_dataloader(True,config=my_data_config)
    model=DNN("Xiamen1_336pre144", dlts, MODULE_CONFIG, TRAIN_CONFIG, my_data_config)
    model.get_trained(by_name=True)
    print(model.more_test())

def fangwu():
    dlts, dlts_config = get_dataloader(False)
    eng = get_engine(dlts, dlts_config, EXP_CONFIG)
    # eng.model.summary()
    data = dlts.get_all_data()[0]
    source = data[data[:,1]>0]
    feature=1
    target=source.copy()
    target[:,feature]=0
    fixed = eng.fix(source, dlts.features[feature], target)
    res = pd.DataFrame()
    if dlts_config['normalise']:
        data = dlts.single_descore(data)
        fixed = dlts.single_descore(fixed)
    for idx, feature in enumerate(dlts.features):
        res[feature + "_ori"] = source[:, idx]
        #res[feature + "_tar"] = target[:, idx]
        res[feature + "_fix"] = fixed[:, idx]
    res.to_csv('fangwu.csv', index=False)
    print(eng.rec_loss,eng.KL_loss)

if __name__ == '__main__':
    fangwu()