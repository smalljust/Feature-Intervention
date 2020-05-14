from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from utils.data_processor import DataLoader
from model.data_analyzer import DataAnalyzer
import keras.backend as K
import keras

import matplotlib.pyplot as plt

import numpy as np

class GAN(DataAnalyzer):
    def __init__(self, data_processor: DataLoader, model_config, train_config, data_config=None):
        super(GAN, self).__init__("GAN", data_processor, model_config, train_config, data_config)
        self.original_dim=len(self.data_processor.features)
        self.latent_dim = self.model_config.get("latent_dim",10)
        self.hidden_dim=self.model_config.get("hidden_dim",10)

        # Following parameter and optimizer set as recommended in paper
        self.model=keras.Model()
        self.critic=keras.Model()
        self.generator=keras.Model()

    def build_model(self):
        copt = Adam(self.train_config.get("dlr",0.0004),0.5)
        gopt = Adam(self.train_config.get("glr",0.0002),0.5)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss='binary_crossentropy',
                            optimizer=copt,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The model  (stacked generator and critic)
        self.model = Model(z, valid)
        self.model.compile(loss='binary_crossentropy',
                              optimizer=gopt,
                              metrics=['accuracy'])

    def build_generator(self):

        model = Sequential()

        model.add(Dense(self.hidden_dim, input_dim=self.latent_dim))
        model.add(LeakyReLU())
        model.add(Dense(self.original_dim))

        #model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Dense(self.hidden_dim,input_dim=self.original_dim))
        model.add(LeakyReLU())
        model.add(Dense(self.latent_dim))
        model.add(LeakyReLU())
        model.add(Dense(1, activation='sigmoid'))
        img = Input(shape=(self.original_dim,))
        validity = model(img)

        return Model(img, validity)

    def train(self):
        batch_size=self.train_config.get("batch_size",128)
        gen_fuc=self.data_processor.generate_train_batch(batch_size)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(self.train_config.get("epoch",4000)):
            noise=np.random.normal(0, 1, (batch_size, self.latent_dim))
            imgs,_=next(gen_fuc)
            gen_imgs=self.generator.predict(noise)
            d_loss_real=self.critic.train_on_batch(imgs,valid)
            d_loss_fake=self.critic.train_on_batch(gen_imgs,fake)
            d_loss=0.5*np.add(d_loss_fake,d_loss_real)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss=self.model.train_on_batch(noise,valid)
            print("%d [D loss: %f, real_acc: %.2f%%, fake_acc: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100 * d_loss_real[1], 100 * d_loss_fake[1], g_loss[0]))
