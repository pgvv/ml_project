import os
import sys
from dataclasses import dataclass

#Algorithms
from catboost import CatBoostRegressor
from sklearn.ensemble import(AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#Logging & Exception Handling
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_filepath= os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            X_train, y_train, X_test, y_test= (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Booster": GradientBoostingRegressor(),
                "K-Neighbours Classifier": KNeighborsRegressor(),
                "CAT Booster": CatBoostRegressor(verbose= False),
                "ADA Booster": AdaBoostRegressor(),
                "XGB Classifier": XGBRegressor()
            }

            model_report: dict= evaluate_model(
                X_train= X_train, 
                y_train= y_train,
                X_test= X_test,
                y_test= y_test,
                models= models)
            
            #To get best model score
            best_model_score= max(sorted(model_report.values()))
            #To get corresponding best model name
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException("No best model found")
            
            logging.info("Solving model successfully found for both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_filepath,
                obj= best_model
            )

            #Results and Metrics
            predict_score= best_model.predict(X_test)
            r2_square= r2_score(y_test, predict_score)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys)