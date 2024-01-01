import tensorflow as tf
from pathlib import Path
import mlflow
from urllib.parse import urlparse
import pandas as pd 
import numpy as np
from sklearn.metrics import (classification_report,precision_score,recall_score,f1_score,)
from bbc_news import logger
from bbc_news.entity import EvaluationConfig
from bbc_news.utils import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        test_x = test_data.drop(['cat'], axis=1)
        test_y = test_data[['cat']]
        
        self.score = self.model.evaluate(test_x,test_y)
        self.save_score()
        
        predicted_qualities = self.model.predict(test_x)
        (self.tst_f1,self.tst_precission,self.tst_recall) = self.classification_performace_matric(test_y, predicted_qualities,'weighted')
        logger.info(
                    "\n Loss:"+ str(self.score[0])+ "\n Accuracy:"+ str(self.score[1]) +
                     "\n F1 score:"+str(self.tst_f1)+"\n Precission:"+str(self.tst_precission) +
                     "\n Recall:"+ str(self.tst_recall)
                     )
        
        
    def save_score(self):
        scores = {"Loss": self.score[0], "Accuracy": self.score[1]}
        save_json(path=Path(self.config.metric_file_name), data=scores)
    
    def classification_performace_matric(self,actual, pred,avg):
        #acc=accuracy_score(actual,pred)
        pred=np.argmax(pred, axis=1)
        actual=np.argmax(actual, axis=1)
        f1=f1_score(actual,pred,average=avg)
        precission=precision_score(actual,pred,average=avg)
        recall=recall_score(actual,pred,average=avg)
        logger.info("\nClassification Report \n"+classification_report(actual,pred))
        
        return f1,precission,recall
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run(run_name='Tokenization-2*LSTM'):
            mlflow.log_params(self.config.all_params.features)
            mlflow.log_params(self.config.all_params.lstm)
            mlflow.log_metrics(
                {"Test Loss": self.score[0], "Test Accuracy": self.score[1]}
            )
            
            mlflow.log_metric("Test F1 Score",self.tst_f1)
            mlflow.log_metric("Test Precision",self.tst_precission)
            mlflow.log_metric("Test Recall",self.tst_recall)
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.tensorflow.log_model(self.model, "model", registered_model_name="Tokenization-2*LSTM_Model")
            else:
                mlflow.tensorflow.log_model(self.model, "lstm_model")
