import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class data_transformation_config:
    prepro_obj_filepath= os.path.join("artifacts", "preprocessor.pkl")

class data_transformation:
    def __init__(self):
        self.data_transformation_config= data_transformation_config()

    #This function is responsible for data transformation
    def get_data_transformer_obj(self):
        try:
            numerical_columns= ['reading_score', 'writing_score']
            categorical_columns= ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= "median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling successfully completed")
            logging.info(f"Numerical columns: {numerical_columns}")

            cat_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("one_hot_encoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean= False))
                ]
            )
            logging.info("Categorical columns encoding successfully completed")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Reading training and testing data is completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj= self.get_data_transformer_obj()

            target_column= "math_score"
            numerical_columns= ['reading_score', 'writing_score']

            input_feature_train_df= train_df.drop(columns=[target_column], axis= 1)
            target_feature_train_df= train_df[target_column]

            input_feature_test_df= test_df.drop(columns=[target_column], axis= 1)
            target_feature_test_df= test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_array= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array= preprocessing_obj.transform(input_feature_test_df)

            train_array= np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array= np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            save_object(
                file_path= self.data_transformation_config.prepro_obj_filepath,
                obj= preprocessing_obj
            )

            logging.info("Saved preprocessing object")

            return(
                train_array,
                test_array,
                self.data_transformation_config.prepro_obj_filepath
            )
        except Exception as e:
            raise CustomException(e, sys)
