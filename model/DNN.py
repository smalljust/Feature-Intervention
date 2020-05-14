from model.data_analyzer import DataAnalyzer
import math
import numpy as np
from numpy import newaxis
from utils.data_processor import DataLoader
from utils.timer import Timer
from keras.layers import Dense, Dropout, LSTM, GRU,Input
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import keras
import os

class DNN(DataAnalyzer):
    def __init__(self,my_id,data_processor:DataLoader,model_config,train_config=None,data_config=None):
        super(DNN,self).__init__(my_id,data_processor,model_config,train_config,data_config)
        self.model=keras.models.Sequential()

    def build_model(self):
        timer = Timer()
        timer.start()

        for layer in self.model_config['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'input':
                self.model.add(Input(shape=(input_dim,)))
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation,input_shape=(input_dim,)))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type']=='gru':
                self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        metrics=list(self.load_metrics(self.model_config['metrics']).values())

        self.model.compile(loss=self.model_config['loss'],
                           optimizer=self.model_config['optimizer'],
                           metrics=metrics)

        print('[Model] Model Compiled')
        timer.stop()

    def set_callbacks(self):
        callbacks = []
        for callback in self.train_config.get('callback',[]):
            monitor = callback['monitor'] if 'monitor' in callback else 'val_loss'
            save_best_only = callback['save_best_only'] if 'save_best_only' in callback else True
            min_delta = callback['min_delta'] if 'min_delta' in callback else 0
            patience = callback['patience'] if 'patience' in callback else 0
            baseline=callback.get('baseline',None)
            if callback['type'] == 'ModelCheckpoint':
                callbacks.append(
                    ModelCheckpoint(filepath=os.path.join(self.SAVE_DIR,self.name+'.h5'), monitor=monitor, save_best_only=save_best_only))
            if callback['type'] == 'EarlyStopping':
                callbacks.append(EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience,restore_best_weights=True,baseline=baseline))
            if callback['type'] == 'TensorBoard':
                callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True))
        return callbacks

    def train(self):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.train_config["epochs"], self.train_config["batch_size"]))
        callbacks = self.set_callbacks()
        x,y=self.data_processor.get_train_data()
        self.model.fit(
            x,y,
            validation_data=self.data_processor.get_test_data(),
            epochs=self.train_config["epochs"],
            batch_size=self.train_config["batch_size"],
            callbacks=callbacks
        )
        print('[Model] Training Completed. Model saved as %s' % self.name)
        timer.stop()

    def train_generator(self):
        self.model.summary()
        timer = Timer()
        timer.start()
        steps_per_epoch = math.ceil(self.data_processor.len_train / self.train_config['batch_size'])
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' %
              (self.train_config['epochs'], self.train_config['batch_size'], steps_per_epoch))

        callbacks = self.set_callbacks()
        self.model.fit_generator(
            self.data_processor.generate_train_batch(batch_size=self.train_config['batch_size']),
            validation_data=self.data_processor.get_valid_data(),
            steps_per_epoch=steps_per_epoch,
            epochs=self.train_config['epochs'],
            callbacks=callbacks,
        )

        print('[Model] Training Completed. Model saved as %s' % self.name)
        timer.stop()

    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        t=Timer()
        t.start()
        predicted = self.model.predict(data,batch_size=4096)
        predicted = np.reshape(predicted, (predicted.size,))
        t.stop()
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        #unused
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        #unused
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted

    def more_test(self):
        '''
        查准 查全
        :return:
        '''
        if self.data_processor.len_test > 0:
            data_x, data_y = self.data_processor.get_test_data()
            pre_y=self.model.predict(data_x,batch_size=4096)
            pre_y=[1.0 if y>=0.5 else 0.0 for y in pre_y]
            TP=0#真阳
            FN=0#假阴，预测为假实际为真
            FP=0#假阳，预测为真实际为假
            TN=0#真阴
            for true_y,pre in zip(data_y,pre_y):
                if(true_y==1 and pre==1):
                    TP+=1
                elif(true_y==0 and pre==1):
                    FN+=1
                elif(true_y==1 and pre==0):
                    FP+=1
                else:
                    TN+=1
            return [TP,FN,FP,TN]
        else:
            return [-1]