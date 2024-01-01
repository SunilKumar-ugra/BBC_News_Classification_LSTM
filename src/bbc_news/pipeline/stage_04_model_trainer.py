from bbc_news.config import ConfigurationManager
from bbc_news.components.model_trainer import ModelTrainer

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
