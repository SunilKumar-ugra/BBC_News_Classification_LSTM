from pathlib import Path
import tensorflow as tf
import pandas as pd
from bbc_news.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.lstm_model_path
        )

    # def train_valid_generator(self):

    #     datagenerator_kwargs = dict(
    #         rescale = 1./255,
    #         validation_split=0.20
    #     )

    #     dataflow_kwargs = dict(
    #         target_size=self.config.params_image_size[:-1],
    #         batch_size=self.config.params_batch_size,
    #         interpolation="bilinear"
    #     )

    #     valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #         **datagenerator_kwargs
    #     )

    #     self.valid_generator = valid_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="validation",
    #         shuffle=False,
    #         **dataflow_kwargs
    #     )

    #     if self.config.params_is_augmentation:
    #         train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #             rotation_range=40,
    #             horizontal_flip=True,
    #             width_shift_range=0.2,
    #             height_shift_range=0.2,
    #             shear_range=0.2,
    #             zoom_range=0.2,
    #             **datagenerator_kwargs
    #         )
    #     else:
    #         train_datagenerator = valid_datagenerator

    #     self.train_generator = train_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="training",
    #         shuffle=True,
    #         **dataflow_kwargs 
    #     )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    # def train(self):
    #     self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
    #     self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

    #     self.model.fit(
    #         self.train_generator,
    #         epochs=self.config.params_epochs,
    #         steps_per_epoch=self.steps_per_epoch,
    #         validation_steps=self.validation_steps,
    #         validation_data=self.valid_generator
    #     )

    #     self.save_model(
    #         path=self.config.trained_model_path,
    #         model=self.model
    #     )
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop(['cat'], axis=1)
        test_x = test_data.drop(['cat'], axis=1)
        train_y = train_data[['cat']]
        test_y = test_data[['cat']]


        self.model.fit(
            train_x,train_y,
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size,
            validation_data=(test_x,test_y)
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )  
        
