import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass                       #creats class variables
#Data Transformationb Testing
from src.components.data_transform import data_transformation
from src.components.data_transform import data_transformation_config
#Model Trainer Testing
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#Data Ingestion needs inputs(where test data, input data, raw data needed to be saved).
#All this info is saved in a class
@dataclass
class data_ingestion_config:
    train_data_path: str= os.path.join("artifacts", "train.csv")        #csv file saved in "artifacts" folder
    test_data_path: str= os.path.join("artifacts", "test.csv")
    raw_data_path: str= os.path.join("artifacts", "data.csv")

class data_ingestion:
    def __init__(self):
        self.ingestion_config= data_ingestion_config()      #above 3 links will be saved in sub-variables

    #code required to read from DBs
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")
        try:
            df= pd.read_csv('notebook\data\stud.csv')       #Insert links of any DB sources here
            logging.info("Read the dataset successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok= True)
            #convert df to csv
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)

            logging.info("Train test split initiated")
            train_set, test_set= train_test_split(df, test_size= 0.2, random_state= 42)
            #convert sets to csv
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header= True)

            logging.info("Data Ingestion successfully completed")

            #Return inputs for data transformation
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

#Testing        
if __name__== "__main__":
    obj= data_ingestion()
    train_data, test_data= obj.initiate_data_ingestion()

    DataTransformation= data_transformation()
    train_array, test_array, _= DataTransformation.initiate_data_transformation(train_data, test_data)

    model_trainer= ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array))