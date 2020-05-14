from keras.layers import Input, Dense, Lambda,Dropout,BatchNormalization,Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam,RMSprop
from utils.data_processor import DataLoader
from model.data_analyzer import DataAnalyzer
import keras.backend as K
import keras
import numpy as np
import os


class SpectralNormalization:
    """层的一个包装，用来加上SN。
    """

    def __init__(self, layer):
        self.layer = layer

    def spectral_norm(self, w, r=5):
        w_shape = K.int_shape(w)
        in_dim = np.prod(w_shape[:-1]).astype(int)
        out_dim = w_shape[-1]
        w = K.reshape(w, (in_dim, out_dim))
        u = K.ones((1, in_dim))
        for i in range(r):
            v = K.l2_normalize(K.dot(u, w))
            u = K.l2_normalize(K.dot(v, K.transpose(w)))
        return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

    def spectral_normalization(self, w):
        return w / self.spectral_norm(w)

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        if not hasattr(self.layer, 'spectral_normalization'):
            if hasattr(self.layer, 'kernel'):
                self.layer.kernel = self.spectral_normalization(self.layer.kernel)
            if hasattr(self.layer, 'gamma'):
                self.layer.gamma = self.spectral_normalization(self.layer.gamma)
            self.layer.spectral_normalization = True
        return self.layer(inputs)

class VAEGAN(DataAnalyzer):
    def __init__(self, data_processor: DataLoader, model_config, train_config, data_config=None):
        super(VAEGAN, self).__init__(model_config.get("name","VAEGAN"), data_processor, model_config, train_config, data_config)
        self.original_size=(len(self.data_processor.features),)
        self.hidden_dim=self.model_config.get('hide_nodes')
        if type(self.hidden_dim) != list:
            self.hidden_dim = [self.hidden_dim]
        self.latent_dim=self.model_config.get('latent_dim')
        self.encoder = keras.Model()
        self.decoder = keras.Model()
        self.vae = keras.Model()
        self.discriminate = keras.Model()
        self.gan=keras.Model()
        self.model = keras.Model()

    def apply_layers(self,layers,tensor):
        res=tensor
        for layer in layers:
            res=layer(res)
        return res

    def build_model(self):
        dopt = RMSprop(self.model_config.get("dlr", 0.004))#,clipvalue=1
        gopt = RMSprop()

        input_tensor = Input(shape=self.original_size)
        hide_tensor=input_tensor
        for idx,hidden_node in enumerate(self.hidden_dim):
            hide_layers = [Dense(hidden_node, name='hide_'+str(idx),),Activation('relu')]
            hide_tensor = self.apply_layers(hide_layers,hide_tensor)

        latent_mean_layers = [Dense(self.latent_dim, name='latent_mean')]
        latent_mean_tensor = self.apply_layers(latent_mean_layers,hide_tensor)  # mu
        latent_log_var_layers = [Dense(self.latent_dim, name='latent_log_var')]
        latent_log_var_tensor = self.apply_layers(latent_log_var_layers,hide_tensor)  # log O^2
        latent_var_tensor=Lambda(K.exp)(latent_log_var_tensor)
        self.encoder=keras.Model(input_tensor,[latent_mean_tensor, latent_var_tensor],name='encoder')

        latent_layer = Lambda(self.sampling, output_shape=(self.latent_dim,), name='latent')
        latent_tensor = latent_layer([latent_mean_tensor, latent_log_var_tensor])

        latent_input_tensor = Input(shape=(self.latent_dim,))
        decoder_hide_tensor = latent_tensor
        latent_tensor2=latent_input_tensor
        for idx,hidden_node in enumerate(self.hidden_dim):
            decoder_hide_layers = [Dense(hidden_node, name='decoder_hide_'+str(idx),),Activation('relu')]
            decoder_hide_tensor = self.apply_layers(decoder_hide_layers,decoder_hide_tensor)
            latent_tensor2 = self.apply_layers(decoder_hide_layers,latent_tensor2)
        decoded_layers = [Dense(self.original_size[0], name='decoder_mean')]
        decoded_tensor = self.apply_layers(decoded_layers, decoder_hide_tensor)
        latent_decoded_tensor=self.apply_layers(decoded_layers,latent_tensor2)
        self.decoder=keras.Model(latent_input_tensor,latent_decoded_tensor,name='decoder')
        self.vae = keras.Model(input_tensor, self.decoder(latent_mean_tensor),name='vae')

        gens = Input(shape=self.original_size)
        if self.model_config.get("ALI", False):
            lats = Input(shape=(self.latent_dim,))
            disc_hide_tensor=Lambda(K.concatenate)([gens,lats])
        else:
            disc_hide_tensor=gens
        for idx,hidden_node in enumerate(self.hidden_dim):
            disc_hide_layers = [SpectralNormalization(Dense(hidden_node//2, name='disc_hide_'+str(idx))), Activation('relu')]
            disc_hide_tensor = self.apply_layers(disc_hide_layers, disc_hide_tensor)
        disc_latent_layers = [SpectralNormalization(Dense(self.latent_dim//2, name='disc_latent')),Activation('relu')]
        disc_latent_tensor = self.apply_layers(disc_latent_layers,disc_hide_tensor)
        disc_layers = [SpectralNormalization(Dense(1,name='disc_out')),Activation('sigmoid')]
        disc_tonsor = self.apply_layers(disc_layers,disc_latent_tensor)
        if self.model_config.get("ALI", False):
            self.discriminate=keras.Model([gens,lats],disc_tonsor,name='discriminate')
        else:
            self.discriminate = keras.Model(gens, disc_tonsor,name='discriminate')
        self.discriminate.compile(dopt, loss="binary_crossentropy", metrics=['accuracy'])

        if self.model_config.get("ALI", False):
            frozen_D = keras.Model([gens, lats], disc_tonsor)
            frozen_D.trainable = False
            valid = frozen_D([decoded_tensor,latent_mean_tensor])
        else:
            frozen_D = keras.Model(gens, disc_tonsor)
            frozen_D.trainable = False
            valid = frozen_D(decoded_tensor)
        self.model = keras.Model(input_tensor, valid,name='VAE_GAN')

        if self.model_config.get("ALI", False):
            fake=frozen_D([latent_decoded_tensor,latent_input_tensor])
        else:
            fake = frozen_D(latent_decoded_tensor)
        self.gan = keras.Model(latent_input_tensor, fake,name='GAN')
        ones = K.ones(shape=(self.train_config['batch_size'], 1), dtype=np.float32)
        self.gan.add_loss(K.mean(K.sum(K.binary_crossentropy(ones, fake), axis=-1)))
        self.gan.compile(gopt)

        # from keras.losses import mean_squared_error
        # xent_loss = K.sum(self.stds*K.square(input_tensor-decoder_mean_tensor), axis=-1)
        # xent_loss=mean_squared_error(input_tensor,decoder_mean_tensor)
        recon_loss = K.sum(K.abs(input_tensor-decoded_tensor), axis=-1)
        kl_loss = self.model_config.get("lambda", 1) * -0.5 * K.sum(
            1 + latent_log_var_tensor - K.abs(latent_mean_tensor) - K.exp(latent_log_var_tensor)
            , axis=-1)#- 2*K.abs(latent_mean_tensor)
        gan_loss = 1*K.sum(K.binary_crossentropy(ones, valid), axis=-1)
        vae_loss = K.mean(recon_loss + kl_loss + gan_loss)
        self.vae.add_loss(K.mean(recon_loss + kl_loss))
        self.vae.compile(gopt)
        self.vae.metrics_names.append('recon_loss')
        self.vae.metrics_tensors.append(K.mean(recon_loss))
        self.vae.metrics_names.append('kl_loss')
        self.vae.metrics_tensors.append(K.mean(kl_loss))

        self.model.add_loss(vae_loss)
        self.model.compile(gopt)
        self.model.metrics_names.append('recon_loss')
        self.model.metrics_tensors.append(K.mean(recon_loss))
        self.model.metrics_names.append('kl_loss')
        self.model.metrics_tensors.append(K.mean(kl_loss))
        self.model.metrics_names.append('gan_loss')
        self.model.metrics_tensors.append(K.mean(gan_loss))

    def sampling(self,latent_args):
        '''
        重参数技巧
        '''
        z_mean_tensor, z_log_var_tensor=latent_args
        epsilon = K.random_normal(shape=K.shape(z_mean_tensor))
        return z_mean_tensor + K.exp(z_log_var_tensor / 2) * epsilon

    def train(self):
        batch_size=self.train_config.get("batch_size",128)
        gen_fuc=self.data_processor.generate_train_batch(batch_size)
        best_loss=100

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        test,_=self.data_processor.get_test_data()

        for epoch in range(self.train_config.get("epochs",4000)):
            ins,_=next(gen_fuc)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gens = self.decoder.predict(noise, batch_size=batch_size, verbose=0)
            if self.model_config.get("ALI",False):
                ins_lat,_ = self.encoder.predict(ins, batch_size=batch_size, verbose=0)
                d_loss_fake = self.discriminate.train_on_batch([gens,noise], fake)
                d_loss_real = self.discriminate.train_on_batch([ins,ins_lat], valid)
            else:
                d_loss_fake = self.discriminate.train_on_batch(gens, fake)
                d_loss_real=self.discriminate.train_on_batch(ins,valid)
            d_loss=0.5*np.add(d_loss_fake,d_loss_real)
            g_loss=self.model.train_on_batch(ins,None)
            #self.gan.train_on_batch(noise, None)
            if epoch % 10 == 9:
                recs = self.vae.predict(test, batch_size=batch_size, verbose=0)
                if self.model_config.get("ALI",False):
                    lat,_=self.encoder.predict(test,batch_size=batch_size)
                    rec_acc=self.discriminate.evaluate([recs,lat],np.ones((recs.shape[0], 1)), batch_size=batch_size, verbose=0)
                else:
                    rec_acc = self.discriminate.evaluate(recs, np.ones((recs.shape[0], 1)), batch_size=batch_size, verbose=0)
                print("%d [G loss: %f, D loss:%f,real_acc: %.2f%%, fake_acc: %.2f%%, rec_acc: %.2f%%, rec: %f, KL: %f, GAN: %f]" %
                      (epoch, g_loss[0], d_loss[0], 100 * d_loss_real[1], 100 * d_loss_fake[1],100*rec_acc[1],g_loss[1],g_loss[2],g_loss[3]))
                if epoch>self.train_config.get("epochs",4000)/2 and g_loss[1]<best_loss:
                    self.model.save(os.path.join(self.SAVE_DIR,self.name+'.h5'))
                    best_loss=g_loss[1]

    def cal_performance(self):
        threshold={}
        test_data=self.data_processor.get_test_data()[0]
        pre_y=self.vae.predict(test_data,batch_size=self.train_config['batch_size'],verbose=0)
        loss=self.vae.evaluate(test_data,batch_size=self.train_config['batch_size'],verbose=0)
        rec_loss=loss[1]
        KL_loss=loss[2]
        for idx in range(len(self.data_processor.features)):
            threshold[self.data_processor.features[idx]]=np.sqrt(sum((test_data[:,idx]-pre_y[:,idx])**2)/len(test_data))
        return threshold,rec_loss,KL_loss

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.discriminate.summary()
        self.vae.summary()
        self.gan.summary()
        self.model.summary()