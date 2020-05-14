import pandas as pd
import matplotlib.pyplot as plt
from utils.data_processor import DataLoader
import numpy as np
from model.data_analyzer import DataAnalyzer
import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from utils.timer import Timer

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoches=[]

    def on_epoch_end(self, epoch, logs={}):
        loss=logs.get('loss')
        if loss<150:
            self.epoches.append(epoch)
            self.losses.append(loss)

class VAE(DataAnalyzer):
    def __init__(self,data_processor:DataLoader,model_config,train_config,data_config=None):
        super(VAE, self).__init__("VAE", data_processor, model_config, train_config, data_config)
        self.encoder=keras.Model()
        self.decoder=keras.Model()
        self.model=keras.Model()
        self.linears=[]
        self.threshold={}

    def sampling(self,latent_args):
        '''
        重参数技巧
        '''
        z_mean_tensor, z_log_var_tensor=latent_args
        epsilon = K.random_normal(shape=K.shape(z_mean_tensor))
        return z_mean_tensor + K.exp(z_log_var_tensor / 2) * epsilon

    def build_model(self):
        original_dim=len(self.data_processor.features)
        input_tensor=Input(shape=(original_dim,))
        hide_layer=Dense(self.model_config["hide_nodes"],name='hide')
        hide_tensor=LeakyReLU()(hide_layer(input_tensor))

        latent_mean_layer = Dense(self.model_config["latent_dim"],name='latent_mean')
        latent_mean_tensor = latent_mean_layer(hide_tensor)#mu
        latent_log_var_layer=Dense(self.model_config["latent_dim"],name='latent_log_var')
        latent_log_var_tensor=latent_log_var_layer(hide_tensor)#log O^2

        latent_layer=Lambda(self.sampling, output_shape=(self.model_config["latent_dim"],),name='latent')
        latent_tensor=latent_layer([latent_mean_tensor,latent_log_var_tensor])

        decoder_hide_layer = Dense(self.model_config["hide_nodes"],name='decoder_hide')
        decoder_hide_tensor = LeakyReLU()(decoder_hide_layer(latent_tensor))
        decoder_mean_layer = Dense(original_dim,name='decoder_mean')
        decoder_mean_tensor = decoder_mean_layer(decoder_hide_tensor)

        self.model=Model(input_tensor,decoder_mean_tensor)
        #from keras.losses import mean_squared_error
        #xent_loss = K.sum(self.stds*K.square(input_tensor-decoder_mean_tensor), axis=-1)
        #xent_loss=mean_squared_error(input_tensor,decoder_mean_tensor)
        xent_loss = K.sum(K.square(input_tensor - decoder_mean_tensor), axis=-1)
        kl_loss = self.model_config.get("lambda",1) * -0.5 * K.sum(1 + latent_log_var_tensor - K.square(latent_mean_tensor) - K.exp(latent_log_var_tensor), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        self.model.add_loss(vae_loss)
        self.model.compile(optimizer=self.model_config['optimizer'])
        latent_var_tensor = Lambda(K.exp)(latent_log_var_tensor)
        self.encoder=Model(input_tensor,[latent_mean_tensor,latent_var_tensor])

        latent_input_tensor=Input(shape=(self.model_config["latent_dim"],))
        latent_decoded_tensor=decoder_hide_layer(latent_input_tensor)
        x_decoded=decoder_mean_layer(latent_decoded_tensor)
        self.decoder=Model(latent_input_tensor,x_decoded)

    def train(self):
        history = LossHistory()
        train_data,_=self.data_processor.get_train_data()
        valid_data,_=self.data_processor.get_valid_data()
        def myfit():
            self.model.fit(train_data,
                           shuffle=True,
                           epochs=self.train_config['epochs'],
                           batch_size=self.train_config['batch_size'],
                           validation_data=(valid_data, None),
                           callbacks=[history,keras.callbacks.TerminateOnNaN(),
                                      keras.callbacks.EarlyStopping(mode='min', patience=500,
                                                                    restore_best_weights=True),
                                      keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=100,mode='min')],
                           verbose=2)
        suc=True
        if (not self.train_config['retrain']):
            try:
                self.model.load_weights('vae_weight.h5')
            except:
                self.build_model()
                suc=False
        if(not(suc) or self.train_config['retrain']):
            myfit()
            loss=self.model.evaluate(valid_data,None,batch_size=1024)
            while np.isnan(loss):
                self.build_model()
                myfit()
                loss = self.model.evaluate(valid_data, None, batch_size=1024)
            plt.show()
            plt.scatter([0],[0])
            plt.plot(history.epoches,history.losses)
            plt.savefig('vae_loss.png')
            self.model.save_weights('vae_weight.h5')
        self.cal_threshold()

    def cal_threshold(self):
        test_data=self.data_processor.get_test_data()[0]
        pre_y=self.model.predict(test_data)
        for idx in range(len(self.data_processor.features)):
            self.threshold[idx]=np.sqrt(sum((test_data[:,idx]-pre_y[:,idx])**2)/len(test_data))

    def summary(self):
        self.model.summary()
        self.encoder.summary()
        self.decoder.summary()

    def cal_lr(self):
        train_data=self.data_processor.get_train_data()[0]
        latent_vector,_ = self.encoder.predict(train_data)
        for idx in range(self.data_processor.features):
            y=train_data[:,idx]
            from sklearn import linear_model
            self.linears.append(linear_model.LinearRegression())
            self.linears[-1].fit(latent_vector,y)

    def fix_lr(self,input:np.ndarray,fix_idx,fix_tgt):
        fix_num = fix_tgt[fix_idx]
        latent,_ = self.encoder.predict(input[np.newaxis,:])[0]
        weight=self.linears[fix_idx].coef_[0]
        delta=fix_num-input[fix_idx]
        fixed_latent = latent + delta / np.dot(weight, weight) * weight
        fixed=self.decoder.predict(fixed_latent[np.newaxis,:])[0]
        return fixed

def show_latent_space(data,encoder,linears):
    encoded=encoder.predict(data)
    np.savetxt('tmp1.csv',encoded)
    x1=np.random.uniform(encoded[:,0].min(),encoded[:,0].max(),size=len(data))
    x2=np.random.uniform(encoded[:,1].min(),encoded[:,1].max(),size=len(data))
    X = np.array([x1,x2]).T
    #print(encoded)
    print(encoded[:,0].min(),encoded[:,0].max(),encoded[:,1].min(),encoded[:,1].max())
    #plt.title(str(col),fontsize=30)
    c1=(1.0,0.0,0.0)
    c2=(0.0,0.0,1.0)
    figure,ax=plt.subplots(len(data.columns),2,figsize=(6*2,len(data.columns)*6))
    for idx,col in enumerate(data):
        minn = data[col].min()
        maxn = data[col].max()
        ax[idx][0].set_title(str(col)+'_vae')
        ax[idx][0].axis([encoded[:,0].min()-1,encoded[:,0].max()+1,encoded[:,1].min()-1,encoded[:,1].max()+1])
        ax[idx][0].scatter(encoded[:, 0], encoded[:, 1],s=5,
                    c=[blend_color(c2,c1,(x - minn) / (maxn - minn)) for x in data[col]])
        ax[idx][1].set_title(str(col)+'_linear')
        ax[idx][1].axis(
            [encoded[:, 0].min() - 1, encoded[:, 0].max() + 1, encoded[:, 1].min() - 1, encoded[:, 1].max() + 1])
        Y=linears[idx].predict(X)
        ax[idx][1].scatter(X[:, 0], X[:, 1], s=5,
                            c=[blend_color(c2, c1,
                                           (a - minn) / (maxn - minn)) for a in Y])
        ax[idx][1].plot([-linears[idx].coef_[0]*10,linears[idx].coef_[0]*10],[-linears[idx].coef_[1]*10,linears[idx].coef_[1]*10],color='g',linewidth=3)
        #plt.savefig("res/"+col+".png")
        #plt.show()
    figure.savefig("res/tot.png")

def blend_color(color1, color2, f):
    f=min(1,f)
    f=max(0,f)
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    r = r1 + (r2 - r1) * f
    g = g1 + (g2 - g1) * f
    b = b1 + (b2 - b1) * f
    return (r, g, b)