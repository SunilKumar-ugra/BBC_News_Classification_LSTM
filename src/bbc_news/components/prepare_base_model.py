from bbc_news.entity import PrepareBaseModelConfig
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from bbc_news import logger
from bbc_news.entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
    @staticmethod
    def prepare_lstm_model(max_words, max_len, embedding_dim, lstm_units,num_classes):
        model = Sequential()
        model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
        #model.add(SpatialDropout1D(0.2))
        model.add(LSTM(units=lstm_units, return_sequences=True))
        model.add(LSTM(units=lstm_units))
        model.add(Dense(units=num_classes, activation='softmax'))
        return model
    
    @staticmethod
    def compile_lstm_model(self,model,l,o,m):
        model.compile(loss=l, optimizer=o, metrics=m)
        self.save_model(path=self.config.lstm_model_path, model=model)
        logger.info("LSTM model created and saved")
        
    
    def build_lstm_model(self):
        m= self.prepare_lstm_model(max_words=self.config.params_max_features,
                                       max_len=self.config.params_max_text_length,
                                       embedding_dim=self.config.params_embedding_dim,
                                       lstm_units=self.config.params_lstm_units,
                                       num_classes=self.config.params_num_category)
        m.summary()
        self.compile_lstm_model(self,model=m,
                                l=self.config.params_loss,
                                m=self.config.params_metrics,
                                o=self.config.params_optimizer)
        
