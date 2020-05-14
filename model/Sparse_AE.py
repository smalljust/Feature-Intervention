import pandas as pd
from utils.data_processor import DataLoader
import numpy as np
from model.data_analyzer import DataAnalyzer
import keras
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from utils.timer import Timer


class SAE(DataAnalyzer):
    def __init__(self,data_processor:DataLoader,model_config,train_config,data_config=None):
        super(SAE, self).__init__("SAE", data_processor, model_config, train_config, data_config)
        self.encoder=keras.Model()
        self.decoder=keras.Model()
        self.model=keras.Model()

    def build_model(self):
        original_dim=len(self.data_processor.features)
        input_tensor=Input(shape=(original_dim,))
        hide_layer=Dense(self.model_config["hide_nodes"],name='hide')
        hide_tensor=ReLU()(hide_layer(input_tensor))

        latent_layer = Dense(self.model_config["latent_dim"],name='latent_mean')
        latent_tensor = latent_layer(hide_tensor)#mu

        decoder_hide_layer = Dense(self.model_config["hide_nodes"],name='decoder_hide')
        decoder_hide_tensor = ReLU()(decoder_hide_layer(latent_tensor))
        decoder_mean_layer = Dense(original_dim,name='decoder_mean')
        decoder_mean_tensor = decoder_mean_layer(decoder_hide_tensor)

        self.model=Model(input_tensor,decoder_mean_tensor)
        xent_loss = K.sum(K.square(input_tensor - decoder_mean_tensor), axis=-1)
        sae_loss = K.mean(xent_loss)
        self.model.add_loss(sae_loss)
        self.model.compile(optimizer=self.model_config['optimizer'])
        self.encoder=Model(input_tensor,latent_tensor)

        latent_input_tensor=Input(shape=(self.model_config["latent_dim"],))
        latent_decoded_tensor=decoder_hide_layer(latent_input_tensor)
        x_decoded=decoder_mean_layer(latent_decoded_tensor)
        self.decoder=Model(latent_input_tensor,x_decoded)

    def train(self):
        train_data,_=self.data_processor.get_train_data()
        valid_data,_=self.data_processor.get_valid_data()
        def myfit():
            self.model.fit(train_data,
                           shuffle=True,
                           epochs=self.train_config['epochs'],
                           batch_size=self.train_config['batch_size'],
                           validation_data=(valid_data, None),
                           callbacks=[keras.callbacks.TerminateOnNaN(),
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
            self.model.save_weights('vae_weight.h5')
        self.cal_threshold()

    def cal_threshold(self):
        threshold = {}
        test_data = self.data_processor.get_test_data()[0]
        pre_y = self.model.predict(test_data)
        for idx in range(len(self.data_processor.features)):
            threshold[self.data_processor.features[idx]] = np.sqrt(
                sum((test_data[:, idx] - pre_y[:, idx]) ** 2) / len(test_data))
        return threshold

    def summary(self):
        self.model.summary()
        self.encoder.summary()
        self.decoder.summary()