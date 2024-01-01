from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    lstm_model_path: Path
    updated_base_model_path: Path
    
    # params_image_size: list
    # params_learning_rate: float
    # params_include_top: bool
    # params_weights: str
    # params_classes: int
     
    params_max_features:int
    params_max_text_length: int
    params_num_category: int
    params_loss:str
    params_optimizer: str
    params_activation: str
    params_dropout: float
    params_embedding_dim: str
    params_metrics:float
    params_learning_rate:float
    params_batch_size:  int
    params_epochs: int
    params_lstm_units:int


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path    
    tokeniazer_path: Path
    params_max_features:int
    params_max_text_length: int

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    lstm_model_path: Path
    
    train_data_path: Path
    test_data_path: Path
    
    params_epochs: int
    params_batch_size: int
   

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model_path: Path
    test_data_path: Path
    all_params: dict
    mlflow_uri: str
    metric_file_name: Path
