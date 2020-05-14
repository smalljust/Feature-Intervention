import numpy as np
import pandas as pd
import keras
from keras.layers import Layer,Input
import keras.backend as K
from explainer.optimizer import myOptimizer
from config import *
from model.VAE_GAN import VAEGAN
from model.Sparse_AE import SAE
from keras.optimizers import RMSprop,adam,SGD
from copy import deepcopy
from explainer.valid_exp import Valid_test

def cal_crossentropy(mu1, var1, mu2, var2):
    """
    计算每一维对交叉熵的贡献：
    0.5(log(var2/var1)+var1/var2+(mu2-mu1)^2/var2-1)
    :param mu1:真实
    :param var1:真实
    :param mu2:拟合
    :param var2:拟合
    :return:
    """
    return 0.5 * np.sum((np.log(var2/var1) + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1),axis=-1)


class MyLayer(Layer):
    def __init__(self, output_dim,init_value,**kwargs):
        self.output_dim = output_dim
        self.init_value=init_value
        super(MyLayer, self).__init__(**kwargs)

    def init_val(self,shape, dtype):
        return K.variable(self.init_value)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert self.output_dim == input_shape[1]
        self.kernel = self.add_weight(name='target',
                                      shape=(None, self.output_dim),
                                      initializer=self.init_val,
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        return self.kernel

    def compute_output_shape(self, input_shape):
        return (self.init_value.shape[0], self.output_dim)

class BestWeight(keras.callbacks.Callback):
    def __init__(self,epochs):#,mu1,var1
        super(BestWeight,self).__init__()
        self.epochs=epochs
        self.best_loss = None
        self.best_weight=None
        #self.mu1=mu1
        #self.var1=var1
        #self.mu=[]
        #self.var=[]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        '''
        mu2=logs.get('mu2')
        for mu in mu2:
            self.mu.append(mu)
        var2 = logs.get('var2')
        for var in var2:
            self.var.append(var)
        '''
        loss=None
        gan_loss=logs.get('gan_loss',None)
        if gan_loss:
            loss=gan_loss
            del logs['gan_loss']
        recon_loss=logs.get('recon_loss',None)
        del logs['recon_loss']
        if loss:
            loss=loss+recon_loss
        else:
            loss=recon_loss
        kl_loss=logs.get('kl_loss',None)
        del logs['kl_loss']
        loss=loss+kl_loss

        #print(kl_loss,cal_crossentropy(self.mu1,self.var1,mu2,var2))

        if epoch >= self.epochs // 3:
            if self.best_loss is None:
                self.best_loss=loss
                self.best_weight = self.model.get_layer(name='target').get_weights()[0]
            else:
                c=loss<self.best_loss
                self.best_loss[c]=loss[c]
                self.best_weight[c]=self.model.get_layer(name='target').get_weights()[0][c]

class VAE_engine():
    def __init__(self,data_loader,data_config,exp_config):
        self.model=VAEGAN(data_loader, VAE_MODEL_CONFIG, VAE_TRAIN_CONFIG, data_config)
        #self.model=SAE(data_loader, VAE_MODEL_CONFIG, VAE_TRAIN_CONFIG, data_config)
        self.data_loader=data_loader
        self.data_config=data_config
        self.model.get_model()
        self.model.encoder.trainable=False
        self.model.decoder.trainable=False
        if 'discriminate' in dir(self.model):
            self.model.discriminate.trainable=False
        self.threshold,self.rec_loss,self.KL_loss=self.model.cal_performance()
        self.exp_config=exp_config
        #self.valid_model=None

    def cal_lr(self,feature_idx):
        train_data=self.data_loader.get_train_data()[0]
        latent_vector,_ = self.model.encoder.predict(train_data, batch_size=1024)
        y=train_data[:,feature_idx]
        from sklearn import linear_model
        linear=linear_model.LinearRegression()
        linear.fit(latent_vector,y)
        return linear

    def linear_core(self,source,fig_tgt,fix_idx):
        """
        只能处理一维数据
        :param source:
        :param fig_tgt:
        :param fix_idx:
        :return:
        """
        latent,_ = self.model.encoder.predict(source, batch_size=1024)
        linear=self.cal_lr(fix_idx)
        weight = linear.coef_/linear.coef_.sum()
        weight=weight[np.newaxis,:]
        delta = fig_tgt[:,fix_idx] - source[:,fix_idx]
        delta = delta[:,np.newaxis]
        fixed_latent = latent + delta * weight
        fixed = self.model.decoder.predict(fixed_latent, batch_size=1024)
        return fixed

    def greedy_core(self,source,fix_tgt,fix_idx):
        """
        每一次向目标方向前进var
        :param source:
        :param fix_tgt:
        :param fix_idx:
        :return:
        """
        fixed=source
        for _ in range(1000):
            mask = np.where(abs(fixed - fix_tgt)[:, fix_idx] >= self.threshold[fix_idx])[0]
            if mask.size == 0:
                break
            wait_data = fixed[mask]
            latents_ori, var_ori = self.model.encoder.predict(wait_data, batch_size=1024)
            wait_data[:, fix_idx] = fix_tgt[mask, fix_idx]
            latents, var = self.model.encoder.predict(wait_data, batch_size=1024)
            for idx1, latent in enumerate(latents):
                for idx2, element in enumerate(latent):
                    delta = element - latents_ori[idx1, idx2]
                    if abs(delta) > var[idx1, idx2] + var_ori[idx1, idx2]:
                        latents[idx1, idx2] = latents_ori[idx1, idx2] + delta / abs(delta) * var_ori[idx1, idx2]
            '''
            all_v = np.concatenate((latents_ori[:, :, np.newaxis], latents[:, :, np.newaxis],
                                var_ori[:, :, np.newaxis], var[:, :, np.newaxis]), axis=-1)
            latents = np.apply_along_axis \
                (lambda x: x[0] + np.sign(x[1] - x[0]) * x[2] if abs(x[1] - x[0]) > x[2] + x[3] else x[1], 2, all_v)
            '''
            fixed[mask] = self.model.decoder.predict(latents, batch_size=1024)
            nanpos = np.where(np.isnan(fixed))[0]
            if (nanpos.size > 0):
                fixed[nanpos] = fix_tgt[nanpos]
        return fixed

    def arg_core(self,fixed,fix_tgt,fix_idx):
        latents_ori, vars_ori = self.model.encoder.predict(fixed, batch_size=1024)

        def f(x, args):
            """
            由平方差函数和交叉熵组成
            两个多维高斯分布的交叉熵
            0.5(log(det(var2)/det(var1))+tr(var2.I*var1)+(mu2-mu1).T*var2.I*(mu2-mu1)-n)
            :param x: (batch_num,ori_dim)
            :param args:
                args[0] alpha
            :return:
            """
            if 'discriminate' in dir(self.model):
                loss = 1 - self.model.discriminate.predict(x, batch_size=1024).reshape((x.shape[0],))
            else:
                loss = np.zeros(shape=x.shape[0], dtype=np.float64)
            loss += (x[:, fix_idx] - fix_tgt[:, fix_idx]) ** 2
            latents, vars = self.model.encoder.predict(x, batch_size=1024)
            cross_entropys = np.zeros(shape=x.shape[0], dtype=np.float64)
            for idx in range(latents.shape[1]):
                cross_entropys += cal_crossentropy(latents_ori[:, idx], vars_ori[:, idx], latents[:, idx],
                                                    vars[:, idx])
            loss += args[0] * cross_entropys
            return loss

        # start=self.model.encoder.predict(fix_tgt, batch_size=1024)
        opt = myOptimizer(f, fix_tgt, self.exp_config.get('alpha', 0.1), cfg=self.exp_config.get('opt_cfg', {}))
        opt.arg_min()
        return opt.best_x

    def cal_loss(self, model, bs, dim, fix_idx, fix_tgt, mu1, var1, target):
        mu2, var2 = self.model.encoder(target)

        gan_loss = None
        if hasattr(self.model, 'discriminate') and self.exp_config.get("GAN_loss", 0) != 0:
            gan = self.model.discriminate(target)
            gan_loss = K.sum(K.log(1 - gan), axis=-1)
            model.add_loss(self.exp_config.get("GAN_loss", 0) * K.sum(gan_loss))

        '''
        rel_loss = None
        valid = None
        if self.valid_model is not None and self.exp_config.get("rel_loss", 1) != 0:
            mask2 = np.zeros((dim, dim - 1), dtype=np.float)
            for i in range(dim):
                if i < fix_idx:
                    mask2[i, i] = 1.0
                if i > fix_idx:
                    mask2[i, i - 1] = 1.0
            target_slice2 = K.dot(target, K.variable(mask2))
            valid = K.reshape(self.valid_model(target_slice2), (bs,))
            rel_loss = K.square(K.variable(fix_tgt[:, fix_idx]) - valid)
            model.add_loss(self.exp_config.get("rel_loss", 1) * K.sum(rel_loss))
        '''

        mask = np.zeros((dim, 1), dtype=np.float)
        mask[fix_idx, 0] = 1.0
        target_slice = K.dot(target, K.variable(mask))
        # target_slice = K.slice(target, [0, fix_idx], [bs, 1])
        recon_loss = K.square(K.reshape(target_slice, (bs,)) - K.variable(fix_tgt[:, fix_idx]))
        model.add_loss(self.exp_config.get("recon_loss", 10) * K.sum(recon_loss))
        kl_loss = 0.5 * K.sum(K.log(var2 / var1) + var1 / var2 + K.square(mu1 - mu2) / var2 -1, axis=-1)
        model.add_loss(self.exp_config.get("kl_loss", 1) * K.sum(kl_loss))
        model.compile(SGD(lr=0.0003))
        if gan_loss is not None:
            model.metrics_names.append('gan_loss')
            model.metrics_tensors.append(self.exp_config.get("GAN_loss", 0) * K.sum(gan_loss))
        #if rel_loss is not None:
        #    model.metrics_names.append('rel_loss')
        #    model.metrics_tensors.append(self.exp_config.get("rel_loss", 1) * K.sum(rel_loss))
        model.metrics_names.extend(['recon_loss', 'kl_loss'])
        model.metrics_tensors.extend([self.exp_config.get("recon_loss", 10) *recon_loss,
                                      self.exp_config.get("kl_loss", 1) * kl_loss])
        #model.metrics_names.extend(['mu2','var2'])
        #model.metrics_tensors.extend([mu2,var2])
    def x_core(self,source,fix_tgt,fix_idx):
        """
        直接搜索x
        :param source:
        :param fix_tgt:
        :param fix_idx:
        :return:
        """
        # K.set_floatx('float64')
        bs = fix_tgt.shape[0]

        dim = source.shape[1]
        ori = Input(batch_shape=(bs, dim))

        # ones = Input(batch_shape=(bs, bs))
        # target = Dense(dim, use_bias=False, kernel_initializer=init_val,name='target')(ones)
        target = MyLayer(dim, init_value=source if self.exp_config.get("src_start",True) else fix_tgt, name='target')(ori)
        model = keras.Model(ori, target)
        mu1, var1 = self.model.encoder(ori)

        self.cal_loss(model, bs, dim, fix_idx, fix_tgt, mu1, var1, target)
        best_weight = BestWeight(self.exp_config["epochs"])
        #losses = model.evaluate(source, batch_size=bs, verbose=0)
        #print(losses)
        model.fit(source, batch_size=bs, epochs=self.exp_config["epochs"], verbose=0,callbacks=[best_weight],shuffle=False)
        '''
        losses=model.evaluate(source,batch_size=bs,verbose=0)
        
        tmp = []
        metrics_names.insert(0,'loss')
        for name, output in zip(metrics_names, losses):
            if name.find('loss') < 0:
                tmp.extend(list(output))
            else:
                print(name, output)
        
        pd.DataFrame(tmp).to_csv('tmp.csv')
        '''
        return best_weight

    def z_core(self,source,fix_tgt,fix_idx):
        ori_latents,ori_vars=self.model.encoder.predict(source, batch_size=1024, verbose=0)
        #print(self.model.encoder.get_layer(name='latent_log_var').get_weights())
        #pd.DataFrame(ori_latents).to_csv("latents.csv",index=False)
        #pd.DataFrame(ori_vars).to_csv("vars.csv",index=False)
        tgt_latents=None
        if not self.exp_config.get('src_start',True):
            tgt_latents,_=self.model.encoder.predict(fix_tgt, batch_size=1024, verbose=0)
        bs = fix_tgt.shape[0]
        dim = source.shape[1]
        z_dim = ori_latents.shape[1]

        # ones = Input(batch_shape=(bs, bs))
        # target = Dense(dim, use_bias=False, kernel_initializer=init_val,name='target')(ones)
        mu1, var1 = Input(batch_shape=(bs, z_dim)), Input(batch_shape=(bs, z_dim))
        target_z = MyLayer(z_dim, init_value=tgt_latents if tgt_latents is not None else ori_latents, name='target')(mu1)
        target=self.model.decoder(target_z)
        model = keras.Model([mu1, var1], target)
        self.cal_loss(model, bs, dim, fix_idx, fix_tgt, mu1, var1, target)
        best_weight = BestWeight(self.exp_config["epochs"])#,ori_latents,ori_vars
        model.fit([ori_latents,ori_vars], batch_size=bs, epochs=self.exp_config["epochs"], verbose=0,callbacks=[best_weight],shuffle=False)
        '''
        losses=model.evaluate(input,batch_size=bs,verbose=0)
        tmp = []
        metrics_names.insert(0,'loss')
        for name, output in zip(metrics_names, losses):
            if name.find('loss') < 0:
                tmp.extend(list(output))
            else:
                print(name, output)
        pd.DataFrame(tmp).to_csv('tmp.csv')
        '''
        return best_weight

    def sparse_col(self,source,fix_tgt,fix_idx):
        ori_latents = self.model.encoder.predict(source, batch_size=1024, verbose=0)
        tgt_latents = None
        if not self.exp_config.get('src_start', True):
            tgt_latents = self.model.encoder.predict(fix_tgt, batch_size=1024, verbose=0)
        bs = fix_tgt.shape[0]
        dim = source.shape[1]
        z_dim = ori_latents.shape[1]
        model_input=Input(batch_shape=(bs, z_dim))
        target_z = MyLayer(z_dim, init_value=tgt_latents if tgt_latents is not None else ori_latents, name='target')(
            model_input)
        target = self.model.decoder(target_z)
        model = keras.Model(model_input, target)
        mask = np.zeros((dim, 1), dtype=np.float)
        mask[fix_idx, 0] = 1.0
        target_slice = K.dot(target, K.variable(mask))
        # target_slice = K.slice(target, [0, fix_idx], [bs, 1])
        recon_loss = K.square(K.reshape(target_slice, (bs,)) - K.variable(fix_tgt[:, fix_idx]))
        model.add_loss(self.exp_config.get("recon_loss", 10) * K.sum(recon_loss))
        kl_loss = K.sum((target_z - ori_latents)**2 , axis=-1)
        model.add_loss(self.exp_config.get("kl_loss", 1) * K.sum(kl_loss))
        model.compile(SGD(lr=0.0003))
        best_weight = BestWeight(self.exp_config["epochs"])
        model.fit(ori_latents, batch_size=bs, epochs=self.exp_config["epochs"], verbose=0,
                  callbacks=[best_weight])
        return best_weight

    def fix(self, source:np.ndarray, feature, fix_tgt):
        # return fix_tgt
        # return self.linear_core(source,fix_tgt,self.data_loader.features.index(feature))
        # return self.model.model.predict(fix_tgt,batch_size=1024)
        #if self.exp_config.get('valid',False):
        #    valid_test=Valid_test(deepcopy(self.data_loader.data),deepcopy(self.data_config),VAL_MODULE_CONFIG,VAL_TRAIN_CONFIG)
        #    self.valid_model=valid_test.get_DA(feature).model
        fix_idx=self.data_loader.features.index(feature)
        if self.exp_config.get("search_x",True):
            best_weight= self.x_core(source,fix_tgt,fix_idx)
            return best_weight.best_weight
        else:
            best_weight= self.z_core(source,fix_tgt,fix_idx)
            # best_weight=self.sparse_col(source,fix_tgt,fix_idx)
            #pd.DataFrame(best_weight.mu).to_csv("mu_tmp.csv",index=False)
            #pd.DataFrame(best_weight.var).to_csv("var_tmp.csv", index=False)
            return self.model.decoder.predict(best_weight.best_weight, batch_size=1024)
        # plot_loss(best_weight)


import matplotlib.pyplot as plt
def plot_loss(best_weight):
    x=len(best_weight.kl_loss)
    plt.plot(range(x),best_weight.recon_loss,c='r')
    plt.plot(range(x),best_weight.kl_loss,c='g')
    plt.plot(range(x),best_weight.gan_loss,c='b')
    plt.plot(range(x),best_weight.rel_loss,c='k')
    print(best_weight.best_epoch)
    plt.show()
