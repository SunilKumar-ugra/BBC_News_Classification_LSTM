from bbc_news.constants import *
from bbc_news.utils import read_yaml, create_directories
from bbc_news.entity import (DataIngestionConfig, PrepareBaseModelConfig,DataTransformationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model
            
            create_directories([config.root_dir])

            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                updated_base_model_path=Path(config.updated_base_model_path),
                lstm_model_path=Path(config.lstm_model_path),
                ################################################################
                # params_image_size=self.params.IMAGE_SIZE,
                # params_learning_rate=self.params.LEARNING_RATE,
                # params_include_top=self.params.INCLUDE_TOP,
                # params_weights=self.params.WEIGHTS,
                # params_classes=self.params.CLASSES
                ################################################################
                params_max_features= self.params.features.MAX_FEATURES,
                params_max_text_length= self.params.features.MAX_TEXT_LENGTH,
                params_num_category= self.params.features.NUM_CATEGORY,
                params_loss= self.params.lstm.LOSS,
                params_optimizer=self.params.lstm.OPTIMIZER,
                params_activation=self.params.lstm.ACTIVATION,
                params_dropout=self.params.lstm.DROPOUT,
                params_embedding_dim=self.params.lstm.EMBEDDING_DIM,
                params_metrics=self.params.lstm.METRICS,
                params_learning_rate=self.params.lstm.LEARNING_RATE,
                params_batch_size=self.params.lstm.BATCH_SIZE,
                params_epochs=self.params.lstm.EPOCHS,
                params_lstm_units=self.params.lstm.UNITS
            )

            return prepare_base_model_config
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,    
            tokeniazer_path=config.tokeniazer_path,
            params_max_features= self.params.features.MAX_FEATURES,
            params_max_text_length= self.params.features.MAX_TEXT_LENGTH,
        )

        return data_transformation_config
