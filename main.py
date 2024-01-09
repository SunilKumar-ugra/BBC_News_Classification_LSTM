from bbc_news import logger
from dotenv import load_dotenv

load_dotenv()

from bbc_news.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

from bbc_news.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise 

from bbc_news.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e 

from bbc_news.pipeline.stage_04_model_trainer import ModelTrainingPipeline
STAGE_NAME = "Model Trainer stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
from bbc_news.pipeline.stage_05_model_evaluation_mlflow import EvaluationPipeline

STAGE_NAME = "Evaluation Stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>>  {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e
