from bbc_news.config import ConfigurationManager
from bbc_news.components.model_trainer import ModelTrainer
from bbc_news import logger
class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = ModelTrainer(config=training_config)
        training.get_base_model()
        #training.train_valid_generator()
        training.train()
        
STAGE_NAME = "Model Trainer stage"
if __name__ =='__main__':
    try: 
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
