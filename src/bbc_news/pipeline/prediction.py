import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from bbc_news import logger
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class PredictionPipeline:
    def __init__(self):
        self.model = load_model(Path('artifacts/model_trainer/model.keras'))
    
    @staticmethod
    def tokenize_and_pad_sequences(text_data,max_text_length):
        tokenizer = joblib.load(Path('artifacts/data_transformation/tokeniazer.joblib'))
        tokenizer.fit_on_texts(text_data)
        sequences = tokenizer.texts_to_sequences(text_data)
        padded_sequences = pad_sequences(sequences, maxlen=max_text_length)
        return padded_sequences
    
    def predict(self,data):
       
        x_new=self.tokenize_and_pad_sequences(data,200)
        
        prediction = np.argmax(self.model.predict(x_new))
        logger.info(prediction)
       

        if(prediction==0):                      # 'business': 0,
            pred="business"
            logger.info(pred)
        elif(prediction==1):                    #  'entertainment': 1
            pred="entertainment"
            logger.info(pred)
        elif(prediction==2):                    #  'politics': 2
            pred="politics"
            logger.info(pred)
        elif(prediction==3):                    #  'sport': 3,
            pred="sport"
            logger.info(pred)
        elif(prediction==4):                    #  'tech': 4
            pred="tech"
            logger.info(pred)
        
        return pred
            
STAGE_NAME = "Prediction stage"
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        data="qpr keeper day heads for preston queens park rangers keeper chris day is set to join preston on a month s loan.  day has been displaced by the arrival of simon royce  who is in his second month on loan from charlton. qpr have also signed italian generoso rossi. r s manager ian holloway said:  some might say it s a risk as he can t be recalled during that month and simon royce can now be recalled by charlton.  but i have other irons in the fire. i have had a  yes  from a couple of others should i need them.   day s rangers contract expires in the summer. meanwhile  holloway is hoping to complete the signing of middlesbrough defender andy davies - either permanently or again on loan - before saturday s match at ipswich. davies impressed during a recent loan spell at loftus road. holloway is also chasing bristol city midfielder tom doherty."
        obj = PredictionPipeline()
        
        obj.predict(data)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e