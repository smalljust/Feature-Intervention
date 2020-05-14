PREDECT_DAYS = 3
INPUT_DAYS = 7
INTERVAL_HOURS = 0.5  # 注意每一个文件都要修改

DATA_CONFIG = {
    "filename": "final.csv",
    "features": "1stFlrSF,2ndFlrSF,GrLivArea,BedroomAbvGr,TotRmsAbvGrd,FullBath,HalfBath,KitchenAbvGr,TotalBsmtSF,"
                "GarageArea,GarageCars,HouseStyle".split(','),
    "labels": [],
    "add_label": False,
    "valid_ratio": 0,
    "test_ratio":0.2,
    "normalise": False,
    "shuffle": True,
    "balance": False,
    "add_rand": False,
}

EXP_CONFIG={
    "repair_level":0.2,
    "all_data":True,
    "valid":False,
    "epochs":10000,
    "src_start":True,#皆可
    "search_x":False,
    "GAN_loss":0,
    "rel_loss":0,
    "recon_loss":3,
    "kl_loss":1,
}

VAE_MODEL_CONFIG={
    "name":"LosA",
    "ALI":True,
    "hide_nodes":15,
    "latent_dim":20,
    'optimizer':'rmsprop',
    "lambda":1,#高斯正则项系数
}

VAE_TRAIN_CONFIG={
    'retrain':False,
    'batch_size':512,
    'epochs':20000,
}

TRAIN_CONFIG = {
    "retrain": False,
    "epochs": 50,
    "batch_size": 128,
    "callback": [
        {
            "type": "ModelCheckpoint",
            "monitor": "val_acc",
            "save_best_only": True
        },
        {
            "type": "EarlyStopping",
            "monitor":"val_acc",
            "patience":2,
            "min_delta": 0
        }
    ]
}
MODULE_CONFIG = {
    "save_dir": "saved_models",
    "name": "Xiamen1_{}pre{}{}".format(DATA_CONFIG["sequence_length"], DATA_CONFIG["predict_length"],
                                        "_rand" if DATA_CONFIG["add_rand"] else ""),
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"],  # ,"fmeasure", "recall", "precision"
    "layers": [
        {
            "type": "gru",
            "neurons": 200,
            "input_timesteps": int(DATA_CONFIG["sequence_length"]),
            "input_dim": len(DATA_CONFIG['features']) + (1 if DATA_CONFIG["add_rand"] else 0) + (
                1 if DATA_CONFIG["add_label"] else 0),
            "return_seq": True
        },
        {
            "type": "dropout",
            "rate": 0.2
        },
        {
            "type": "gru",
            "neurons": 100,
            "return_seq": True
        },
        {
            "type": "gru",
            "neurons": 100,
            "return_seq": False
        },
        {
            "type": "dropout",
            "rate": 0.2
        },
        {
            "type": "dense",
            "neurons": 1,
            "activation": "sigmoid"
        }
    ]
}
VAL_TRAIN_CONFIG = {
    "retrain": False,
    "epochs": 2000,
    "batch_size": 256,
    "callback":[
        {
            "type": "EarlyStopping",
            "monitor":"val_mean_absolute_error",
            "patience":10,
            "min_delta": 0,
        },
        {
            "type": "ModelCheckpoint",
            "monitor": "val_mean_absolute_error",
            "save_best_only": True
        },
    ],
}
VAL_MODULE_CONFIG = {
    "by_name":False,
    "loss": "mse",
    "optimizer": "adam",
    "metrics": ["mae"],  # ,"fmeasure", "recall", "precision"
    "layers": [
        {
            "input_dim": len(DATA_CONFIG['features'])-1 + (1 if DATA_CONFIG["add_rand"] else 0) + (
                1 if DATA_CONFIG["add_label"] else 0),
            "type": "dense",
            "neurons": 20,
        },
        {
            "type": "dropout",
            "rate": 0.2
        },
        {
            "type": "dense",
            "neurons": 11,
        },
        {
            "type": "dense",
            "neurons": 1,
            "activation": "linear"
        }
    ]
}
